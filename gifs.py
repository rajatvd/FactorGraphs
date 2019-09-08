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
        pass


# %%
class FakeReshape(Scene):
    def construct(self):
        pass


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
