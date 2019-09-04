"""Computations with factor graphs."""
import numpy as np
import networkx as nx
import string

# %% TODO doc this


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
def map_axes(f, new_axes, fg):
    for edge in fg.edges(f, keys=True):
        old_axis = fg.edges[edge]['axis']
        fg.edges[edge]['axis'] = new_axes.index(old_axis)

# %% TODO: fill up these stubs and add more for basic computations


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
    new_fg = fg.copy()

    edges = fg[f][v]
    factor = fg.nodes[f]['factor']

    # find which axes correspond to the multiedges between f and v
    axes_to_remove = []
    for edge in list(edges.values()):
        axes_to_remove.append(edge['axis'])

    # remove all but the lowest axis, and find where the remaining axes end up
    axes = list(range(factor.ndim))
    [axes.remove(x) for x in axes_to_remove[1:]]

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

    if remove_points:
        new_fg.edges[f, v, key_left].pop('points', None)

    return new_fg


# # %%
# factors = {
#     'A': np.random.randn(2, 10, 20, 10, 10, 3)
# }
# einpath = 'ijkjjl->ijkl'
# fg = factor_graph(factors, einpath)
# fg.edges['A', 'j', 0]['points'] = [(1, 1)]
# expected = np.einsum(einpath, factors['A'])
# expected.shape
# # %%
# f, v = 'A', 'j'
# newfg = combine_multiedges('A', 'j', fg)
# newfg['A'], fg['A']
#
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
    pass

# %%
