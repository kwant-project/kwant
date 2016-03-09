# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from kwant.graph import Graph
from kwant.graph.utils import (make_undirected, remove_duplicates,
                               induced_subgraph)
from kwant.graph.defs import gint_dtype

def test_make_undirected():
    graph = Graph(True)
    graph.add_edge(0, 1)
    graph.add_edge(1, 0)
    graph.add_edge(1, 2)
    graph.add_edge(2, -1)
    g = graph.compressed()

    # First, test with no duplicates removed,
    g2 = make_undirected(g, remove_dups=False)

    assert g2.num_nodes == g.num_nodes
    assert g2.num_edges == 6
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)
    assert g2.has_edge(2, 1)

    # then with duplicates removed,
    g2 = make_undirected(g, remove_dups=True)

    assert g2.num_nodes == g.num_nodes
    assert g2.num_edges == 4
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)
    assert g2.has_edge(2, 1)

    # and finally with weights.
    g2, edge_w2 = make_undirected(g, remove_dups=True, calc_weights=True)

    assert g2.num_nodes == g.num_nodes
    assert g2.num_edges == 4
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)
    assert g2.has_edge(2, 1)
    assert edge_w2[g2.first_edge_id(0,1)] == 2
    assert edge_w2[g2.first_edge_id(1,0)] == 2
    assert edge_w2[g2.first_edge_id(1,2)] == 1
    assert edge_w2[g2.first_edge_id(2,1)] == 1

def test_remove_duplicates():
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    # First test without edge weights,
    g = graph.compressed()
    remove_duplicates(g)
    assert g.num_edges == 2
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 2)

    # then with edge weights.
    g = graph.compressed()
    edge_w = np.array([1,1,1], dtype=gint_dtype)
    remove_duplicates(g, edge_w)
    assert g.num_edges == 2
    assert g.has_edge(0, 1)
    assert g.has_edge(1, 2)
    assert edge_w[g.first_edge_id(0,1)] == 2
    assert edge_w[g.first_edge_id(1,2)] == 1


def test_induced_subgraph():
    num_nodes = 6

    graph = Graph()
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1)
    graph.add_edge(1, 0)
    g = graph.compressed()

    # First test select array,
    select = np.array([True, True, True, False, False, True])
    g2 = induced_subgraph(g, select)
    assert g2.num_nodes == 4
    assert g2.num_edges == 3
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)

    # then test select function.
    g2 = induced_subgraph(g, lambda i: select[i])
    assert g2.num_nodes == 4
    assert g2.num_edges == 3
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)

    # Now the same with edge weights.
    edge_w = np.arange(g.num_edges, dtype=gint_dtype)
    g2, edge_w2 = induced_subgraph(g, select, edge_w)
    assert g2.num_nodes == 4
    assert g2.num_edges == 3
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)
    assert edge_w[g.first_edge_id(0,1)] == edge_w2[g2.first_edge_id(0,1)]
    assert edge_w[g.first_edge_id(1,0)] == edge_w2[g2.first_edge_id(1,0)]
    assert edge_w[g.first_edge_id(1,2)] == edge_w2[g2.first_edge_id(1,2)]

    g2, edge_w2 = induced_subgraph(g, lambda i: select[i], edge_w)
    assert g2.num_nodes == 4
    assert g2.num_edges == 3
    assert g2.has_edge(0, 1)
    assert g2.has_edge(1, 0)
    assert g2.has_edge(1, 2)
    assert edge_w[g.first_edge_id(0,1)] == edge_w2[g2.first_edge_id(0,1)]
    assert edge_w[g.first_edge_id(1,0)] == edge_w2[g2.first_edge_id(1,0)]
    assert edge_w[g.first_edge_id(1,2)] == edge_w2[g2.first_edge_id(1,2)]
