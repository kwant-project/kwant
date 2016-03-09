# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from kwant.graph import Graph
# from kwant.graph.scotch import bisect, reset

def _DISABLED_test_bisect():
    # REMARK: This test is somewhat limited in the sense that it can only test
    #         for the general sanity of the output, not if the bisection is
    #         really good (balanced, etc.).
    size = 5
    graph = Graph()

    for i in range(size - 1):
        offset = i * size
        for j in range(size - 1):
            graph.add_edge(offset + j, offset + j + 1)
            graph.add_edge(offset + j + 1, offset + j)
        if i > 0:
            for j in range(size):
                graph.add_edge(offset + j, offset + j - size)
                graph.add_edge(offset + j - size, offset + j)
    g = graph.compressed()

    parts = bisect(g)
    for i in range(g.num_nodes):
        assert parts[i] == 0 or parts[i] == 1

def _DISABLED_test_reset():
    size = 5
    graph = Graph()

    for i in range(size - 1):
        offset = i * size
        for j in range(size - 1):
            graph.add_edge(offset + j, offset + j + 1)
            graph.add_edge(offset + j + 1, offset + j)
        if i > 0:
            for j in range(size):
                graph.add_edge(offset + j, offset + j - size)
                graph.add_edge(offset + j - size, offset + j)
    g = graph.compressed()

    # After calling reset, SCOTCH returns identical results.
    reset()
    parts1 = bisect(g)
    reset()
    parts2 = bisect(g)

    assert (parts1 == parts2).all()
