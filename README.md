# Animating Factor Graphs

This repo contains the code for my blog post on [Visualizing Tensor Operations with Factor Graphs](https://rajatvd.github.io/Factor-Graphs/).

The code uses [`manimnx`](https://github.com/rajatvd/manim-nx), a package to help interface [`manim`](https://github.com/3b1b/manim) with [`networkx`](https://networkx.github.io) to create the animations.


* `factor_graph.py` contains code which purely deals with computations and representing factor graphs using networkx.
* `fg_anim.py` has a few helper functions specific to animating factor graphs.
* `gifs.py` has code for each of the gifs/animations in the post.
