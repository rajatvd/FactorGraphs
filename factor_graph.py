"""Computations with factor graphs."""
import numpy as np
import networkx as nx
import string


# %%
def factor_graph(factors, einpath):
    """Create a factor graph as a networkx MultiDiGraph.

    A MultiDiGraph is used as opposed to just a MultiGraph to enforce
    consistency. Edges are always factor -> variable. This property is
    maintained even when nodes are contracted using nx.contracted_nodes and can
    be relied on for further functionality of code. Note that this is not
    needed from a mathematical point of view, and is purely for convenient
    coding.

    Parameters
    ----------
    factors : dict
        Dict mapping factor name to the numpy array containing the factor data.
    einpath : str
        A valid einstein summation which describes the relations between factors
        and implicitly defines variables as well. This is used to build the
        graph.

    Returns
    -------
    nx.MultiDiGraph
        The factor graph.

    """
    fg = nx.MultiDiGraph()  # make graph
    for f_name, f in factors.items():  # add factors
        fg.add_node(f_name, factor=f, type='factor')

    einpath = einpath.replace(' ', '')
    lhs, rhs = einpath.split("->")

    # add variables
    variables = set([c for c in einpath if c in string.ascii_lowercase])
    for v in variables:
        fg.add_node(v, summed=v not in rhs, type='variable')

    facts = lhs.split(",")
    for f_name, vs in zip(factors.keys(), facts):
        f = factors[f_name]
        for i, v in enumerate(vs):
            size = f.shape[i]
            if 'size' in fg.nodes[v].keys():
                assert fg.nodes[v]['size'] == size, "invalid einpath"
            else:
                fg.nodes[v]['size'] = size

            fg.add_edge(f_name, v, size=size, axis=i)

    return fg


# %%
def map_axes(f, old_axes, fg):
    """Reassign the axis attribute of edges from the factor f.

    Parameters
    ----------
    f : node key
        The factor to change axes of.
    old_axes : list
        old_axis[i] has the old axis attribute of the edge which will be
        assigned axis i.
    fg : MultiDiGraph
        The factor graph.

    Returns
    -------
    None

    """
    assert fg.nodes[f]['type'] == 'factor', "Can't map axes of variable node"
    for edge in fg.edges(f, keys=True):
        old_axis = fg.edges[edge]['axis']
        fg.edges[edge]['axis'] = old_axes.index(old_axis)


# %%
def kill_multiedges_reaxis(f, v, fg):
    """Combine all multiedges between f and v in fg.

    DOES NOT PERFORM COMPUTATION. This is a helper function and should only be
    used in other functions which perform the appropriate computation.

    No indexing is performed on the factor of f. This purely deals with the
    edges in the factor graph and the axis attributes of the corresponding
    edges.

    The combination of edges in the graph is separate from the actual
    computation because actual products which might result in multiedges do not
    separately perform an indexing as that is extremely inefficient.
    (See combine_factors for an example.)

    Parameters
    ----------
    f : node key
        The factor.
    v : node key
        The variable.
    fg : MultiDiGraph
        The factor graph.

    Returns
    -------
    MultiDiGraph
        Copy of the factor graph with multiedges killed.

    """
    new_fg = fg.copy()
    edges = fg[f][v]
    if len(edges) == 1:
        return new_fg
    factor = fg.nodes[f]['factor']

    # find which axes correspond to the multiedges between f and v
    axes_to_remove = []
    for edge in list(edges.values()):
        axes_to_remove.append(edge['axis'])

    # remove all but the lowest axis, and find where the remaining axes end up
    axes = list(range(factor.ndim))
    [axes.remove(x) for x in axes_to_remove[1:]]

    # remove all edges except the lowest axis one in the new copy
    keys_to_remove = []
    for key, data in edges.items():
        if data['axis'] != axes_to_remove[0]:
            keys_to_remove.append(key)
            new_fg.remove_edge(f, v, key)
        else:
            key_left = key

    new_fg.edges[f, v, key_left]['contraction'] = {(f, v, i): fg.edges[f, v, i]
                                                   for i in keys_to_remove}

    # update axes
    map_axes(f, axes, new_fg)
    return new_fg


# %%
def combine_multiedges(f, v, fg, remove_points=True):
    """Combine all multiedges between the factor f and variable v.

    fg should be a valid factor graph (which is a MultiDiGraph).

    Performs a diag-style indexing of the factor f along all axes
    which have an edge to variable v.

    Removes all multiedges except the one with smallest axis.

    Parameters
    ----------
    remove_points: bool
        Whether to remove any points in the resulting combined edge.

    Returns
    -------
    Graph
        A copy of the graph with the multiedges combined, and computations
        performed.

    """
    new_fg = kill_multiedges_reaxis(f, v, fg)

    edges = fg[f][v]
    factor = fg.nodes[f]['factor']

    # find which axes correspond to the multiedges between f and v
    axes_to_remove = []
    for edge in list(edges.values()):
        axes_to_remove.append(edge['axis'])

    # actually perform the diag computation using einsum
    # source is the left part of the einpath
    source = list(string.ascii_lowercase[:factor.ndim])
    for i in axes_to_remove:
        source[i] = 'z'

    # target is the right part of the einpath
    target = []
    for c in source:
        if c not in target:
            target.append(c)

    # compute and update factor in new copy
    einpath = ''.join(source) + '->' + ''.join(target)
    new_factor = np.einsum(einpath, factor)
    new_fg.nodes[f]['factor'] = new_factor

    if remove_points:
        list(new_fg[f][v].values())[0].pop('points', None)

    return new_fg


# # %%
# A = np.random.randn(20, 10, 20, 10, 10, 3)
# B = np.random.randn(10, 3, 20)
# factors = {
#     'A': A,
#     'B': B
# }
# einpath = 'ijijjk,jki -> ijk'
# fg = factor_graph(factors, einpath)
# f1 = 'A'
# f2 = 'B'
# # fg.edges['A', 'j', 0]['points'] = [(1, 1)]
# # expected = np.einsum(einpath, factors['A'])
# # expected.shape
# # %%
# c = np.einsum('ijijjk,jki -> ijk', A, B)
# c.shape
# # %%
# f, v = 'B', 'l'
# newfg = combine_multiedges('A', 'j', fg)
# newfg['A'], fg['A']
#
# # %%


# %%
def compute_sum(v, fg):
    """Compute the sum of a variable v already marked to be summed.

    Sum over the given variable and update the factor connected
    to it. (akin to marginalization)

    Note that the variable v must have only one edge to a factor, so that the
    sum is forced to be a simple linear order computation in the size of the
    variable, and does not actually involve any network contraction.


    Returns
    -------
    Graph
        A copy of the graph with the variable and edge removed, and the factor
        appropriately marginalized.

    """
    assert fg.nodes[v]['summed'] == True, "variable must be summed to compute"
    assert fg.degree(v) == 1, "variable must have only one edge"

    new_fg = fg.copy()

    # compute the sum
    f = list(fg.predecessors(v))[0]
    old_factor = fg.nodes[f]['factor']
    axis = list(fg[f][v].values())[0]['axis']
    new_factor = old_factor.sum(axis=axis)

    # calculate cost of this summation
    m, n = old_factor.reshape(-1, old_factor.shape[axis]).shape
    additions = m*(n-1)

    new_fg.remove_node(v)
    new_fg.nodes[f]['factor'] = new_factor

    return new_fg


# %%
def combine_variables(v1, v2, fg, multiedges=True):
    """Combine the two variables v1 and v2 into v1.

    No computation is actually performed, unless combine_multiedges
    is set to true.

    Parameters
    ----------
    v1, v2 : node keys
        The variables to combine.
    fg : MultiDiGraph
        The factor graph containing the variables.
    multiedges: bool
        Whether to persist (True) any resulting multiedges due to variable
        combination into a single edge, or just combine them (False). Default
        is to persist them (True).

    Returns
    -------
    MultiDiGraph
        Copy of the factor graph with variables combined.

    """
    new_fg = nx.contracted_nodes(fg, v1, v2)
    new_fg2 = new_fg.copy()
    if not multiedges:
        for f in new_fg.predecessors(v1):
            new_fg2 = combine_multiedges(f, v1, new_fg2)

    return new_fg2


# %%
def combine_factors(f1, f2, fg, multiedges=True):
    """Combine the two factors f1 and f2 into f1.

    A broadcasted hadamard product is performed. Note that if multiedges are
    persisted, and then separately combined, it is much more inefficient than
    combining them directly in this function as a huge number of wasted
    multiplications will be computed using the former approach.

    To see why, consider two vectors u and v, and the following computation:

    diag(uv^T)

    This is a diag of an outer product. Ideally, we only need to perform n
    multiplications to get the diagonal elements of the outer product, but if
    we separate the computation into two steps -- first the outer product
    followed by the diag indexing, we perform n^2 multiplications, which is
    much worse.

    Parameters
    ----------
    f1, f2 : node keys
        The factors to combine.
    fg : MultiDiGraph
        The factor graph containing the variables.
    multiedges: bool
        Whether to persist (True) any multiedges after factor combination, or
        just combine them (False). Default is to persist them (True). In the
        case of combining, the indexing is efficiently done to prevent wasted
        multiplications. Note that any multiedges which previously existed
        are also removed.

    Returns
    -------
    MultiDiGraph
        Copy of the factor graph with factors combined.

    """
    new_fg = fg.copy()

    a = fg.nodes[f1]['factor']
    b = fg.nodes[f2]['factor']

    axis_increase = a.ndim  # ndim of f1's factor
    for edge in new_fg.edges(f2, keys=True):
        new_fg.edges[edge]['axis'] += axis_increase

    new_fg = nx.contracted_nodes(new_fg, f1, f2)

    if multiedges:
        # compute outer product without any indexing
        c = a.reshape(*a.shape, *[1 for _ in b.shape]) * b
        new_fg.nodes[f1]['factor'] = c
        return new_fg
    else:
        indices1 = [0]*fg.degree(f1)
        for edge in fg.edges(f1, keys=True):
            indices1[fg.edges[edge]['axis']] = edge[1]

        indices2 = [0]*fg.degree(f2)
        for edge in fg.edges(f2, keys=True):
            indices2[fg.edges[edge]['axis']] = edge[1]

        assert 0 not in indices1+indices2, "Axes of the factors are not valid,\
        the factor graph is invalid"

        target = []
        for ind in indices1+indices2:
            if ind not in target:
                target.append(ind)

        # efficiently compute outer product with indexing
        einpath = ''.join(indices1)+','+''.join(indices2)+'->'+''.join(target)
        c = np.einsum(einpath, a, b)

        # need a factor with the full ndim for kill_multiedges_reaxis to work
        new_fg.nodes[f1]['factor'] = a.reshape(*a.shape, *[1 for _ in b.shape])

        # combine the multiedges in the graph appropriately
        new_fg2 = new_fg.copy()
        for v in new_fg.successors(f1):
            new_fg2 = kill_multiedges_reaxis(f1, v, new_fg2)

        # assign real factor in the end
        new_fg2.nodes[f1]['factor'] = c

        return new_fg2
