# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from nose.tools import assert_true
import numpy as np
# from kwant.graph import Graph, dissection
# from kwant.graph.dissection import edge_dissection

def _DISABLED_test_edge_dissection():
    # REMARK: This test is somewhat limited in the sense that it can only test
    #         for the general sanity of the output, not if the disection is
    #         really good (balanced, etc.).  Sanity is checked by making sure
    #         that every node is included exactly once in the tree.
    size = 5
    graph = Graph()

    for i in xrange(size - 1):
        offset = i * size
        for j in xrange(size - 1):
            graph.add_edge(offset + j, offset + j + 1)
            graph.add_edge(offset + j + 1, offset + j)
        if i > 0:
            for j in xrange(size):
                graph.add_edge(offset + j, offset + j - size)
                graph.add_edge(offset + j - size, offset + j)
    g = graph.compressed()

    tree = edge_dissection(g, 1)
    found = np.zeros(g.num_nodes, dtype = int)

    def parse_tree(entry):
        if type(entry) is tuple:
            parse_tree(entry[0])
            parse_tree(entry[1])
        else:
            found[entry] += 1

    parse_tree(tree)
    assert_true((found == 1).all())
