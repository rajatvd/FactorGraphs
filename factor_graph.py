from manimlib.imports import *
import numpy as np
import networkx as nx
from manimnx import manimnx as mnx
import string

# %%


def factor_graph(factors, einpath):
    fg = nx.Graph()  # make graph

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
    nname = n
    n = G.node[n]
    type_to_shape = {
        'variable': Circle,
        'factor': Square
    }

    def node_color(node):
        if node['type'] == 'factor':
            return GREEN
        return WHITE if not node['summed'] else GRAY
    bg = type_to_shape[n['type']](color=BLACK, fill_color=node_color(n),
                                  fill_opacity=1, radius=0.3, side_length=0.8)
    node_name = TextMobject(nname, color=BLACK)
    x, y = n['pos']
    node_name.move_to(x*RIGHT + y*UP)
    bg.move_to(x*RIGHT + y*UP)
    return VGroup(bg, node_name)


# %%
def get_fg_edge(n1, n2, G):
    s = G[n1][n2]['size']
    n1 = G.node[n1]
    n2 = G.node[n2]
    x1, y1 = n1['pos']
    x2, y2 = n2['pos']
    start = x1*RIGHT + y1*UP
    end = x2*RIGHT + y2*UP
    return Line(start, end, stroke_width=s/10, color=BLACK)


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
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'E': E,
        }

        einpath = 'xu,uy,uv,zwv,y -> xyzw'

        fg = factor_graph(factors, einpath)

        mfg = mnx.ManimGraph(fg, get_fg_node, get_fg_edge)

        self.play(ShowCreation(mfg))
        self.wait(2)
