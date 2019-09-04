"""Animations related to graph transforms"""
from manimlib.imports import *
import networkx as nx
from manimnx import manimnx as mnx
from itertools import chain

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


# %% TODO: take in custom transform for edges and nodes
def transform_graph(mng, G):
    """Transforms the graph in ManimGraph mng to the graph G.

    For any new mob_ids in G, or any nodes or edges without a mob_id, they
    are assumed to be new objects which are to be created. If they contain
    an expansion attribute, they will be created and transformed from those
    mob_ids in the expansion. Otherwise, they will be faded in.

    Missing mob_ids in G are dealt with in two ways:
    If they are found in the contraction attribute of the node or edge, the
    mobjects are first transformed to the contracted node then faded away. If
    not found in any contraction, they are faded away. (Note that this means
    that any older contractions present in G do not affect the animation in any
    way.)

    Cannot contract to new nodes/edges and expand from new nodes/edges.
    # TODO: consider changing this behavior


    Parameters
    ----------
    mng : ManimGraph
        The ManimGraph object to transform.
    G : Graph
        The graph containing attributes of the target graph.

    Returns
    -------
    list of Animations
        List of animations to show the transform.

    """
    anims = []
    id_to_mobj = {**mng.nodes, **mng.edges}

    old_ids = list(mng.nodes.keys()) + list(mng.edges.keys())
    new_ids = []

    # just copying the loops for edges and nodes, not worth functioning that
    # i think
    # ------- ADDITIONS AND DIRECT TRANSFORMS ---------
    # NODES
    for node, node_data in G.nodes.items():
        new_node = mng.get_node(node, G)
        if 'mob_id' not in node_data.keys():
            G.nodes[node]['mob_id'] = mng.count
            mng.count += 1

        mob_id = node_data['mob_id']
        new_ids.append(mob_id)

        if mob_id in old_ids:
            # if mng.graph.nodes[mng.id_to_node[mob_id]] != node_data:
            anims.append(Transform(mng.nodes[mob_id], new_node))
        else:
            if 'expansion' in node_data.keys():
                objs = [id_to_mobj[o['mob_id']]
                        for o in node_data['expansion'].values()]
                anims.append(TransformFromCopy(VGroup(*objs), new_node))
            else:
                anims.append(FadeIn(new_node))

            mng.nodes[mob_id] = new_node
            mng.add(new_node)
            mng.id_to_node[mob_id] = node

    # EDGES
    for edge, edge_data in G.edges.items():
        new_edge = mng.get_edge(edge, G)
        if 'mob_id' not in edge_data.keys():
            G.edges[edge]['mob_id'] = mng.count
            mng.count += 1

        mob_id = edge_data['mob_id']
        new_ids.append(mob_id)

        if mob_id in old_ids:
            # only transform if new is different from old
            # TODO: how to check this properly, and is it really needed?

            # if mng.graph.edges[mng.id_to_edge[mob_id]] != edge_data:
            anims.append(Transform(mng.edges[mob_id], new_edge))
        else:
            if 'expansion' in edge_data.keys():
                objs = [id_to_mobj[o['mob_id']]
                        for o in edge_data['expansion'].values()]
                anims.append(TransformFromCopy(VGroup(*objs), new_edge))
            else:
                anims.append(FadeIn(new_edge))

            mng.edges[mob_id] = new_edge
            mng.add_to_back(new_edge)
            mng.id_to_edge[mob_id] = edge

    # --------- REMOVALS AND CONTRACTIONS ----------
    # NODES
    for node, node_data in mng.graph.nodes.items():
        mob_id = node_data['mob_id']
        if mob_id in new_ids:
            continue

        contracts_to = []
        for node2, node_data2 in G.nodes.items():
            for c2 in node_data2.get('contraction', {}).values():
                if mob_id == c2['mob_id']:
                    contracts_to.append(id_to_mobj[node_data2['mob_id']])
                    break

        mobj = id_to_mobj[mob_id]
        if len(contracts_to) == 0:
            anims.append(FadeOut(mobj))
        else:
            anims.append(Succession(Transform(mobj, VGroup(*contracts_to)),
                                    FadeOut(mobj)))

        # dont actually remove so that the edges in the back remain in the
        # back while fading. this might bite later though
        # mng.remove(mobj)
        del mng.nodes[mob_id]
        del mng.id_to_node[mob_id]

    # EDGES
    for edge, edge_data in mng.graph.edges.items():
        mob_id = edge_data['mob_id']
        if mob_id in new_ids:
            continue

        contracts_to = []
        for edge2, edge_data2 in G.edges.items():
            for c2 in edge_data2.get('contraction', {}).values():
                if mob_id == c2['mob_id']:
                    contracts_to.append(id_to_mobj[edge_data2['mob_id']])
                    break

        mobj = id_to_mobj[mob_id]
        if len(contracts_to) == 0:
            anims.append(FadeOut(mobj))
        else:
            anims.append(Succession(Transform(mobj, VGroup(*contracts_to)),
                                    FadeOut(mobj)))

        # mng.remove(mobj)
        del mng.edges[mob_id]
        del mng.id_to_edge[mob_id]

    mng.graph = G
    return anims


# %%
class Test(Scene):
    def construct(self):
        import sys
        sys.path.append('.')
        from factor_graph import factor_graph
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

        a1 = factor_graph(factors, 'ij,kl,mn -> ijklmn')
        for node, pos in pre_pos.items():
            a1.nodes[node]['pos'] = pos

        mfg1 = mnx.ManimGraph(a1, get_fg_node, get_fg_edge_curve)
        self.play(ShowCreation(mfg1))

        a2 = a1.copy()
        a2.add_node('D', type='factor', factor=C, pos=(0, 0), expansion={
                    'A': a1.nodes['A'], ('A', 'i', 0): a1.edges[('A', 'i', 0)]})
        a2.add_node('E', type='factor', factor=C, pos=(0, 1))
        self.play(*transform_graph(mfg1, a2))

        a3 = a2.copy()
        a3.add_node('f', type='variable', pos=(4, 1), summed='False')
        a3.add_edge('D', 'f', size=100)
        self.play(*transform_graph(mfg1, a3))

        a4 = a3.copy()
        a4.nodes['D']['pos'] = (5, 3)
        a4.remove_node('E')
        self.play(*transform_graph(mfg1, a4))

        a5 = nx.contracted_nodes(a4, 'D', 'A')
        a5.remove_node('B')
        self.play(*transform_graph(mfg1, a5))

        a6 = nx.contracted_nodes(a5, 'C', 'D')
        self.play(*transform_graph(mfg1, a6))

        a7 = a6.copy()
        a7.nodes['C']['pos'] = (0, 0)
        self.play(*transform_graph(mfg1, a7))
        self.wait(2)
