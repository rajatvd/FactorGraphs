"""Animations related to graph transforms"""
from manimlib.imports import *
import networkx as nx
from manimnx import manimnx as mnx
from itertools import chain
transform_graph = mnx.transform_graph

# %%


def to_3d(pos):
    """Convert 2d coords to 3d numpy arrays by adding a 0 in the z coordinate.

    Converts already 3d coords to numpy arrays.

    Parameters
    ----------
    pos : tuple, list, ndarray
        The coords to be converted to 3d.

    Returns
    -------
    numpy array
        3d coords.

    """
    if len(pos) == 3:
        return np.array(pos)
    return np.array(list(pos)+[0])

# %%


def get_fg_node(n, G):
    """Get a mobject for node n in the factor graph G.

    Green square for factors and white/gray circle for variables depending on
    whether they are summed or not. The name of the node is also present in the
    shape.

    Parameters
    ----------
    n : node key
        Node key in factor graph G for which to get the mobject.
    G : MultiDiGraph
        The factor graph.

    Returns
    -------
    VGroup
        The square/circle mobject along with the name inside it.

    """
    node = G.node[n]
    type_to_shape = {
        'variable': Circle,
        'factor': Square
    }

    def node_color(node_dict):
        if node_dict['type'] == 'factor':
            return GREEN
        return WHITE if not node_dict['summed'] else GRAY

    bg = type_to_shape[node['type']](color=BLACK, fill_color=node_color(node),
                                     fill_opacity=1, radius=0.3,
                                     side_length=0.8)
    node_name = TextMobject(n, color=BLACK)
    x, y = node['pos']
    grp = VGroup(bg, node_name)
    grp.move_to(x*RIGHT + y*UP)
    return grp

# %%


def get_fg_edge_curve(ed, G):
    """Get a mobject for edge ed in the factor graph G.

    A smooth curve between the start and end nodes of the edge, passing through
    the points in the 'points' attribute of the edge. The stroke width is
    proportional to the sqrt of the size of variable which this edge is
    connected to.

    Parameters
    ----------
    n : edge key
        Edge key in factor graph G for which to get the mobject.
    G : MultiDiGraph
        The factor graph.

    Returns
    -------
    VMobject
        The bezier curve VMobject for the edge.

    """
    s = G.edges[ed]['size']
    node1 = G.node[ed[0]]
    node2 = G.node[ed[1]]
    x1, y1 = node1['pos']
    x2, y2 = node2['pos']
    start = x1*RIGHT + y1*UP
    end = x2*RIGHT + y2*UP

    # TODO: add default points for multiedges

    pnts = [x*RIGHT + y*UP for x, y in G.edges[ed].get('points', [])]

    edge = VMobject(stroke_width=np.sqrt(s)+1, color=BLACK)
    edge.set_points_smoothly([start, *pnts, end])

    return edge

# %%


# %%
class Test(Scene):
    def construct(self):
        import sys
        sys.path.append('.')
        from factor_graph import factor_graph, combine_multiedges, compute_sum
        np.random.seed()

        A = np.random.randn(30, 30)
        B = np.random.randn(30, 30)
        C = np.random.randn(30, 30)
        D = np.random.randn(10, 10, 10)
        E = np.random.randn(20)
        factors = {
            'A': A,
            'B': B,
            'C': C,
        }

        pre_pos = {
            'A': (-4, -3),
            'i': (-2, -1),
            'j': (-2, -3),
            'B': (4, -3),
            'k': (2, -3),
            'l': (2, -2),
            'C': (0, 3),
            'm': (-1, 2),
            'n': (1, 2),
        }

        fg = factor_graph(factors, 'ij,kl,mn -> ijklmn')
        for node, pos in pre_pos.items():
            fg.nodes[node]['pos'] = pos

        fg.edges['A', 'i', 0]['points'] = [(-3, -2)]

        mfg1 = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)
        self.play(ShowCreation(mfg1))

        fg = fg.copy()
        fg.add_node('D', type='factor', factor=C, pos=(0, 0), expansion={
                    'A': fg.nodes['A'], ('A', 'i', 0): fg.edges[('A', 'i', 0)]})
        fg.add_node('E', type='factor', factor=C, pos=(0, 1))
        self.play(*transform_graph(mfg1, fg))

        fg = fg.copy()
        fg.add_node('f', type='variable', pos=(4, 1), summed='False')
        fg.add_edge('D', 'f', size=100)
        self.play(*transform_graph(mfg1, fg))

        fg = fg.copy()
        fg.nodes['D']['pos'] = (5, 3)
        fg.remove_node('E')
        self.play(*transform_graph(mfg1, fg))

        fg = nx.contracted_nodes(fg, 'j', 'i')
        fg.remove_node('B')
        self.play(*transform_graph(mfg1, fg))

        fg = nx.contracted_nodes(fg, 'C', 'D')
        self.play(*transform_graph(mfg1, fg))

        fg = fg.copy()
        fg.nodes['C']['pos'] = (0, 0)
        self.play(*transform_graph(mfg1, fg))

        fg = combine_multiedges('A', 'j', fg)
        self.play(*transform_graph(mfg1, fg))

        fg.nodes['j']['summed'] = True
        self.play(*transform_graph(mfg1, fg))

        fg = compute_sum('j', fg)
        self.play(*transform_graph(mfg1, fg))

        print(fg.nodes['A']['factor'], np.trace(A))

        self.wait(2)
