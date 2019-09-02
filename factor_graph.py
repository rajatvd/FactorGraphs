"""Computations with factor graphs."""
import numpy as np
import networkx as nx
import string

# %% TODO doc this


def factor_graph(factors, einpath):
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


# %% TODO: fill up these stubs and add more for basic computations

def combine_multiedges(f, v, fg):
    """Combine all multiedges between the factor f and variable v.

    fg should be a valid factor graph (which is a MultiDiGraph).

    Performs a diag-style indexing of the factor f along all axes
    which have an edge to variable v.

    Returns
    -------
    Graph
        A copy of the graph with the multiedges combined, and computations
        performed.

    """

    pass


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
