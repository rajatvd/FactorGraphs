"""Animations related to graph transforms"""
from manimlib.imports import *

import networkx as nx
from manimnx import manimnx as mnx
from itertools import chain

# %% TODO doc this


def to_3d(pos):
    return np.array(list(pos)+[0])

# %% TODO doc this


def get_fg_node(n, G):
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

# %% TODO doc this


def get_fg_edge_curve(ed, G):
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

        mng.remove(mobj)
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

        mng.remove(mobj)
        del mng.edges[mob_id]
        del mng.id_to_edge[mob_id]

    mng.graph = G
    return anims


# %% TODO: fix this up
class Test(Scene):
    def construct(self):
        np.random.seed()
        A = np.random.randn(1000, 100)
        B = np.random.randn(100, 20)
        C = np.random.randn(100, 100)
        D = np.random.randn(100, 100, 100)
        E = np.random.randn(20)
        factors = {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'E': E,
        }

        # einpath = 'iik -> ik'
        einpath = 'xu,uy,uv,zwv,y -> xyzw'

        fg = factor_graph(factors, einpath)
        mfg = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.play(ShowCreation(mfg))
        self.wait(2)
        self.play(*combine_nodes('u', 'v', mfg))
        self.wait(2)
