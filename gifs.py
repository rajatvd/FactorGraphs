from manimlib.imports import *
from manimnx import manimnx as mnx
import numpy as np
import networkx as nx
import sys
import sympy.geometry as sp
sys.path.append('.')
if True:
    from factor_graph import *
    from fg_anim import *


# %%
class MatVec(Scene):
    def construct(self):

        A = np.random.randn(30, 60)
        B = np.random.randn(60)
        factors = {'A': A, 'B': B}
        fg = factor_graph(factors, 'ij,k->ijk')

        h = 2.5
        pos = np.zeros((5, 2))
        pos[:, 0] = (np.arange(5)-2)*h
        mnx.map_attr('pos', ['i', 'A', 'j', 'k', 'B'], pos, fg)

        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        fg = combine_variables('j', 'k', fg)

        pos = np.zeros((4, 2))
        pos[:, 0] = (np.arange(4)-1.5)*h
        mnx.map_attr('pos', ['i', 'A', 'j', 'B'], pos, fg)
        self.play(*mnx.transform_graph(mng, fg))

        fg = fg.copy()
        fg.nodes['j']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg))

        self.wait(2)


# %%
class MatMul(Scene):
    def construct(self):

        A = np.random.randn(30, 60)
        B = np.random.randn(60, 10)
        factors = {'A': A, 'B': B}
        fg = factor_graph(factors, 'ij,kl->ijkl')

        h = 2
        n = 6
        pos = np.zeros((n, 2))
        pos[:, 0] = (np.arange(n)-n/2.0 + 0.5)*h
        mnx.map_attr('pos', ['i', 'A', 'j', 'k', 'B', 'l'], pos, fg)

        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        fg = combine_variables('j', 'k', fg)

        pos = np.zeros((n-1, 2))
        pos[:, 0] = (np.arange(n-1)-(n-1)/2.0+0.5)*h
        mnx.map_attr('pos', ['i', 'A', 'j', 'B', 'l'], pos, fg)
        self.play(*mnx.transform_graph(mng, fg))

        fg = fg.copy()
        fg.nodes['j']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg))

        self.wait(2)


# %%
class Hadamard(Scene):
    def construct(self):
        A = np.random.randn(10, 60)
        B = np.random.randn(10, 60)
        factors = {'A': A, 'B': B}
        fg = factor_graph(factors, 'ij,kl->ijkl')

        posA = np.zeros((3, 2))
        posA[:, 0] = -2
        posA[:, 1] = np.array([2, 0, -2])
        mnx.map_attr('pos', ['i', 'A', 'j'], posA, fg)

        posB = np.zeros((3, 2))
        posB[:, 0] = 2
        posB[:, 1] = np.array([2, 0, -2])
        mnx.map_attr('pos', ['k', 'B', 'l'], posB, fg)

        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        fg = combine_variables('i', 'k', fg)
        fg = combine_variables('j', 'l', fg)
        fg.nodes['i']['pos'] = (0, 2)
        fg.nodes['j']['pos'] = (0, -2)
        fg.edges['A', 'i', 0]['points'] = [(-1.5, 1)]
        fg.edges['A', 'j', 0]['points'] = [(-1.5, -1)]
        fg.edges['B', 'i', 0]['points'] = [(1.5, 1)]
        fg.edges['B', 'j', 0]['points'] = [(1.5, -1)]

        self.play(*mnx.transform_graph(mng, fg))

        self.wait(2)

# %%


class Reshape(Scene):
    def construct(self):
        A = np.random.randn(100)
        factors = {'A': A}
        fg = factor_graph(factors, 'i->i')

        fg.nodes['A']['pos'] = (-1, 0)
        fg.nodes['i']['pos'] = (1, 0)
        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        A = np.random.randn(20, 5)
        factors = {'A': A}
        fg2 = factor_graph(factors, 'ij->ij')
        fg2.nodes['A']['pos'] = (-1, 0)
        fg2.nodes['i']['pos'] = (1, 1)
        fg2.nodes['j']['pos'] = (1, -1)
        fg2.nodes['A']['mob_id'] = fg.nodes['A']['mob_id']
        fg2.nodes['i']['mob_id'] = fg.nodes['i']['mob_id']
        fg2.nodes['j']['expansion'] = {'i': fg.nodes['i']}
        fg2.edges['A', 'i', 0]['mob_id'] = fg.edges['A', 'i', 0]['mob_id']
        fg2.edges['A', 'j', 0]['expansion'] = {
            ('A', 'i', 0): fg.edges['A', 'i', 0]}

        self.play(*mnx.transform_graph(mng, fg2))

        self.wait(2)


# %%
class ReshapeCombine(Scene):
    def construct(self):
        A = np.random.randn(20, 5)
        factors = {'A': A}
        fg = factor_graph(factors, 'ij->ij')

        fg.nodes['A']['pos'] = (-1, 0)
        fg.nodes['i']['pos'] = (1, 1)
        fg.nodes['j']['pos'] = (1, -1)
        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        fg2 = fg.copy()
        fg2.nodes['i']['pos'] = (1, 0.3)
        fg2.nodes['j']['pos'] = (1, -0.3)
        fg2.edges['A', 'i', 0]['points'] = [(0.7, 0)]
        fg2.edges['A', 'j', 0]['points'] = [(0.7, 0)]
        self.play(*mnx.transform_graph(mng, fg2))

        self.wait(2)


# %%
class Trace(Scene):
    def construct(self):
        A = np.random.randn(30, 30)
        factors = {'A': A}
        fg = factor_graph(factors, 'ij->ij')
        fg.nodes['A']['pos'] = (0, 0)
        fg.nodes['i']['pos'] = (0, 2)
        fg.nodes['j']['pos'] = (0, -2)

        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        self.add(mng)
        self.wait(2)

        fg = combine_variables('i', 'j', fg)
        fg.nodes['A']['pos'] = (-1, 0)
        fg.nodes['i']['pos'] = (1, 0)
        fg.edges['A', 'i', 0]['points'] = [(0, 1)]
        fg.edges['A', 'i', 1]['points'] = [(0, -1)]

        self.play(*mnx.transform_graph(mng, fg))
        self.wait(1)

        eq1 = TexMobject(r"\text{diag}(A)", color=BLACK)
        eq2 = TexMobject(r"\text{tr}(A)", color=BLACK)
        eq1.move_to(2*UP)
        eq2.move_to(2*UP)

        self.play(FadeIn(eq1))
        self.wait(1)

        fg.nodes['i']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg), Transform(eq1, eq2))

        self.wait(4)


# %%
class HexTransform(Transform):
    CONFIG = {
        "path_arc": -np.pi/1.5,
    }


# %%
class TraceCyclic(Scene):
    def construct(self):
        A = np.random.randn(30, 60)
        B = np.random.randn(60, 10)
        C = np.random.randn(10, 30)
        factors = {'A': A, 'B': B, 'C': C}
        fg = factor_graph(factors, 'ij,kl,mn->ijklmn')

        tA = TexMobject('A', color=BLACK)
        tC = TexMobject('C', color=BLACK)

        eq = TexMobject(r"\text{tr}(", 'A', 'B', 'C', ')', color=BLACK)
        eq.to_edge(UP)
        dfact = 0.6
        eq.shift(dfact*DOWN)
        tB = eq.submobjects[2]
        tA.to_edge(UP)
        tA.shift(4.5*LEFT+dfact*DOWN)

        tC.to_edge(UP)
        tC.shift(4.5*RIGHT+dfact*DOWN)

        self.add(tA, tB, tC)

        h = 1.5
        n = 9
        pos = np.zeros((n, 2))
        pos[:, 1] = 1
        pos[:, 0] = (np.arange(n)-n/2.0 + 0.5)*h
        mnx.map_attr('pos', ['i', 'A', 'j', 'k', 'B', 'l', 'm', 'C', 'n'], pos, fg)

        mng = mnx.ManimGraph(fg, get_fg_node, get_fg_edge_curve)

        delta = tB.get_center()[0] - fg.nodes['B']['pos'][0]
        eq.shift(delta*LEFT)

        self.add(mng)
        self.wait(2)

        fg = combine_variables('j', 'k', fg)
        mnx.shift_nodes(['A', 'i', 'j'], np.array([h, 0]), fg)
        self.play(*mnx.transform_graph(mng, fg),
                  ReplacementTransform(tA, eq.submobjects[1]))

        fg = fg.copy()
        fg.nodes['j']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg))
        self.wait(1)

        fg = combine_variables('m', 'l', fg)
        mnx.shift_nodes(['C', 'm', 'n'], np.array([-h, 0]), fg)
        self.play(*mnx.transform_graph(mng, fg),
                  ReplacementTransform(tC, eq.submobjects[3]))

        fg = fg.copy()
        fg.nodes['m']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg))
        self.wait(1)

        fg = combine_variables('i', 'n', fg)
        fg.nodes['i']['pos'] = (0, -1)
        fg.edges['A', 'i', 0]['points'] = [(pos[0][0], 0.7),
                                           (pos[0][0], -0.6)]
        fg.edges['C', 'i', 0]['points'] = [(pos[-1][0], 0.7),
                                           (pos[-1][0], -0.6)]
        self.play(*mnx.transform_graph(mng, fg))

        print(fg.edges['A', 'i', 0]['points'],
              fg.edges['C', 'i', 0]['points'],
              fg.nodes['A']['pos'],
              fg.nodes['C']['pos'],)

        fg = fg.copy()
        fg.nodes['i']['summed'] = True
        self.play(*mnx.transform_graph(mng, fg),
                  FadeIn(eq.submobjects[0]), FadeIn(eq.submobjects[-1]))
        self.wait(1)

        eq2 = TexMobject(r"\text{tr}(ABC)", r"= \text{tr}(CAB)", color=BLACK)
        eq2.to_edge(UP)
        eq2.shift(dfact*DOWN)

        eq3 = TexMobject(r"\text{tr}(ABC) = \text{tr}(CAB)",
                         r"= \text{tr}(BCA)", color=BLACK)
        eq3.to_edge(UP)
        eq3.shift(dfact*DOWN)

        nodes = ['m', 'B', 'j', 'A', 'i', 'C']
        edges = [
            ('B', 'm', 0),
            ('B', 'j', 0),
            ('A', 'j', 0),
            ('A', 'i', 0),
            ('C', 'i', 0),
            ('C', 'm', 0),
        ]

        hex = sp.RegularPolygon(sp.Point2D(0, -1), r=2.5, n=12)
        poses = [np.array(p.evalf()).astype(np.float64) for p in hex.vertices]

        mnx.map_attr('pos', nodes, poses[::2], fg)
        mnx.map_attr('points', edges, [[p*0.95] for p in poses[1::2]], fg)
        self.play(*mnx.transform_graph(mng, fg))
        self.wait(2)

        def rotate(l, n):
            return l[-n:] + l[:-n]

        poses = rotate(poses, 4)
        mnx.map_attr('pos', nodes, poses[::2], fg)
        mnx.map_attr('points', edges, [[p*0.95] for p in poses[1::2]], fg)
        self.play(*mnx.transform_graph(mng, fg,
                                       node_transform=HexTransform,
                                       edge_transform=HexTransform,))

        self.play(ApplyMethod(eq.move_to, eq2.submobjects[0].get_center()),
                  FadeIn(eq2.submobjects[1]))
        self.add(eq2.submobjects[0])
        self.remove(eq)
        self.wait(1)

        poses = rotate(poses, 4)
        mnx.map_attr('pos', nodes, poses[::2], fg)
        mnx.map_attr('points', edges, [[p*0.95] for p in poses[1::2]], fg)
        self.play(*mnx.transform_graph(mng, fg,
                                       node_transform=HexTransform,
                                       edge_transform=HexTransform,))
        self.play(ApplyMethod(eq2.move_to,
                              eq3.submobjects[0].get_center()),
                  FadeIn(eq3.submobjects[1]))
        self.wait(4)


# %%
class MatrixInnerProduct(Scene):
    def construct(self):
        pass


# %%
class KronProperty(Scene):
    def construct(self):
        pass


# %%
class SVD(Scene):
    def construct(self):
        pass
