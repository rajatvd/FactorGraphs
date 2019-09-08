from manimlib.imports import *
from manimnx import manimnx as mnx
import numpy as np
import networkx as nx
import sys
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
class TraceCyclic(Scene):
    def construct(self):
        pass


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
