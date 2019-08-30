from manimlib.imports import *
import numpy as np
import networkx as nx
from manimnx import manimnx as mnx
import string

# %%


def factor_graph(factors, einpath):
    fg = nx.MultiGraph()  # make graph

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
            if 'size' in fg[v].keys():
                assert fg[v]['size'] == size
            else:
                fg.nodes[v]['size'] = size

            fg.add_edge(f_name, v, size=size)

    return fg


# %%


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
                                     fill_opacity=1, radius=0.3, side_length=0.8)
    node_name = TextMobject(n, color=BLACK)
    x, y = node['pos']
    grp = VGroup(bg, node_name)
    grp.move_to(x*RIGHT + y*UP)
    return grp

# %%


def get_fg_edge_line(n1, n2, G):
    s = G[n1][n2]['size']
    node1 = G.node[n1]
    node2 = G.node[n2]
    x1, y1 = node1['pos']
    x2, y2 = node2['pos']
    start = x1*RIGHT + y1*UP
    end = x2*RIGHT + y2*UP
    edge = Line(start, end, stroke_width=s/10, color=BLACK)
    return edge

# %%


def get_closest_polygonal_point(n1, pos, G):
    deg = G.degree[n1]
    node = G.node[n1]
    phase = node.get('phase', 0)
    rad = node.get('radius', 0.4)
    pos = np.array(pos)
    centre = np.array(node['pos'])
    thetas = np.arange(deg) * 2*np.pi/deg + phase
    polyg = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    points = centre + polyg*rad
    dists = np.linalg.norm(points-pos, axis=1)
    return np.array(list(points[dists.argmin()])+[0])

# %%


def get_fg_edge_polygonal(n1, n2, G):
    s = G[n1][n2]['size']
    node1 = G.node[n1]
    node2 = G.node[n2]
    x1, y1 = node1['pos']
    x2, y2 = node2['pos']
    start = x1*RIGHT + y1*UP
    end = x2*RIGHT + y2*UP

    p1 = get_closest_polygonal_point(n1, (x2, y2), G)
    p2 = get_closest_polygonal_point(n2, (x1, y1), G)

    edge = VMobject(stroke_width=s/10, color=BLACK)
    edge.set_points_smoothly([start, p1, p2, end])

    # Line(start, end, stroke_width=s/10, color=BLACK)

    return edge

# %%


def get_fg_edge_curve(n1, n2, k, G):
    s = G[n1][n2][k]['size']
    node1 = G.node[n1]
    node2 = G.node[n2]
    x1, y1 = node1['pos']
    x2, y2 = node2['pos']
    start = x1*RIGHT + y1*UP
    end = x2*RIGHT + y2*UP

    pnts = [x*RIGHT + y*UP for x, y in G[n1][n2][k].get('points', [])]

    edge = VMobject(stroke_width=s/10, color=BLACK)
    edge.set_points_smoothly([start, *pnts, end])

    return edge

# %%


class Test(Scene):
    def construct(self):
        np.random.seed()
        A = np.random.randn(10, 100)
        B = np.random.randn(100, 20)
        C = np.random.randn(100, 50)
        D = np.random.randn(10, 100, 50)
        E = np.random.randn(20)
        factors = {
            # 'A': A,
            # 'B': B,
            # 'C': C,
            'D': D,
            # 'E': E,
        }

        einpath = 'iik -> ik'

        fg = factor_graph(factors, einpath)

        mfg = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.play(ShowCreation(mfg))
        self.wait(2)
