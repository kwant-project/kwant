# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import pickle
from io import StringIO
from itertools import zip_longest
import numpy as np
from pytest import raises
from kwant.graph.core import (Graph, NodeDoesNotExistError,
                              EdgeDoesNotExistError, DisabledFeatureError)

def test_empty():
    graph = Graph()
    g = graph.compressed()
    assert not g.twoway
    assert not g.edge_nr_translation
    raises(NodeDoesNotExistError, g.out_neighbors, 0)
    raises(NodeDoesNotExistError, g.has_edge, 0, 0)
    raises(DisabledFeatureError, g.edge_id, 0)
    g = graph.compressed(twoway=True, edge_nr_translation=True)
    assert g.twoway
    assert g.edge_nr_translation
    raises(NodeDoesNotExistError, g.in_neighbors, 0)
    raises(NodeDoesNotExistError, g.out_edge_ids, 0)
    raises(NodeDoesNotExistError, g.in_edge_ids, 0)
    raises(EdgeDoesNotExistError, g.edge_id, 0)

def test_num_nodes():
    graph = Graph()
    assert graph.num_nodes == 0
    graph.num_nodes = 2
    assert graph.num_nodes == 2
    raises(ValueError, graph.__setattr__, 'num_nodes', 1)
    g = graph.compressed()
    assert g.num_nodes == 2

def test_large():
    num_edges = 1000
    graph = Graph()
    for i in range(num_edges):
        graph.add_edge(i, i + 1)
    g = graph.compressed()
    g2 = graph.compressed(twoway=True)
    assert num_edges == g.num_nodes - 1
    for i in range(num_edges):
        assert tuple(g.out_neighbors(i)) == (i + 1,)
        assert tuple(g2.in_neighbors(i + 1)) == (i,)

def check_dot(dot_expect, graph):
    output = StringIO()
    graph.write_dot(output)
    assert output.getvalue() == dot_expect
    output.close()

def test_small():
    g = Graph()
    edges = [(0, 1), (0, 2), (1, 2), (2, 1)]
    for edge in edges:
        g.add_edge(*edge)
    dot_expect = """digraph g {
  0 -> 1;
  0 -> 2;
  1 -> 2;
  2 -> 1;
}
"""
    check_dot(dot_expect, g)
    g = g.compressed(twoway=True)

    for edge_should, edge_is in zip_longest(edges, g):
        assert edge_should == edge_is

    edge_ids = []
    for edge in edges:
        edge_ids.append(g.first_edge_id(*edge))

    assert tuple(g.out_neighbors(0)) == (1, 2)
    assert tuple(g.in_neighbors(0)) == ()
    assert tuple(g.out_edge_ids(0)) == (edge_ids[0], edge_ids[1])
    assert tuple(g.in_edge_ids(0)) == ()

    assert tuple(g.out_neighbors(1)) == (2,)
    assert tuple(g.in_neighbors(1)) == (0, 2)
    assert tuple(g.out_edge_ids(1)) == (edge_ids[2],)
    assert tuple(g.in_edge_ids(1)) == (edge_ids[0], edge_ids[3])

    assert tuple(g.out_neighbors(2)) == (1,)
    assert tuple(g.in_neighbors(2)) == (0, 1)
    assert tuple(g.out_edge_ids(2)) == (edge_ids[3],)
    assert tuple(g.in_edge_ids(2)) == (edge_ids[1], edge_ids[2])

    assert g.has_edge(0, 1)
    g.first_edge_id(0, 1)
    assert not g.has_edge(1, 0)
    raises(IndexError, g.first_edge_id, 1, 0)
    check_dot(dot_expect, g)

def test_negative_node_ids():
    g = Graph()
    raises(ValueError, g.add_edge, 0, -1)

    g = Graph(allow_negative_nodes=True)
    g.add_edge(0, -1)
    g.add_edge(-2, 0)
    raises(ValueError, g.add_edge, -3, -4)
    raises(ValueError, g.compressed)
    g1 = g.compressed(allow_lost_edges=True)
    assert g1.num_px_edges == 1
    assert g1.num_xp_edges == 0
    assert g1.has_edge(0, -1)
    raises(DisabledFeatureError, g1.has_edge, -2, 0)
    assert tuple(g1.out_neighbors(0)) == (-1,)
    g2 = g.compressed(twoway=True)
    assert g2.num_px_edges == 1
    assert g2.num_xp_edges == 1
    assert g2.has_edge(0, -1)
    assert g2.has_edge(-2, 0)
    assert tuple(g2.out_neighbors(0)) == (-1,)
    assert tuple(g2.in_neighbors(0)) == (-2,)

def test_add_edges():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 4), (2, 4), (3, 4)]

    def fill0(g):
        for edge in edges:
            g.add_edge(*edge)
    def fill1(g):
        g.add_edges(edges)
    def fill2(g):
        g.add_edges(np.array(edges))

    prev_dot = None
    for fill in [fill0, fill1, fill2]:
        g = Graph()
        fill(g)
        g = g.compressed()
        output = StringIO()
        g.write_dot(output)
        dot = output.getvalue()
        if prev_dot is not None:
            assert dot == prev_dot
        prev_dot = dot

def test_edge_ids():
    gr = Graph(allow_negative_nodes=True)
    edges = [(0, -1), (-1, 0), (1, 2), (1, 2), (0, -1), (-1, 0), (-1, 0)]
    for edge_nr, edge in enumerate(edges):
        assert gr.add_edge(*edge) == edge_nr

    g = gr.compressed(twoway=True, edge_nr_translation=True)
    assert g.twoway
    assert g.edge_nr_translation
    assert sorted(g.out_edge_ids(1)) == sorted(g.in_edge_ids(2))
    for edge_id in g.out_edge_ids(1):
        assert g.tail(edge_id) == 1
        assert g.head(edge_id) == 2
    for i, edge_id in enumerate(g.all_edge_ids(0, -1)):
        if i == 0:
            assert edge_id == g.first_edge_id(0, -1)
        assert g.tail(edge_id) == 0
        assert g.head(edge_id) == -1
    assert i == 1
    for i, edge_id in enumerate(g.all_edge_ids(-1, 0)):
        if i == 0:
            assert edge_id == g.first_edge_id(-1, 0)
        assert g.tail(edge_id) == None
        assert g.head(edge_id) == 0
    assert i == 2

    for edge_nr, edge in enumerate(edges):
        if edge[0] < 0: continue
        edge_id = g.edge_id(edge_nr)
        assert edge == (g.tail(edge_id), g.head(edge_id))

    g = gr.compressed(edge_nr_translation=True, allow_lost_edges=True)
    raises(EdgeDoesNotExistError, g.edge_id, 1)


def test_pickle():
    gr = Graph(allow_negative_nodes=True)
    edges = [(0, -1), (-1, 0), (1, 2), (1, 2), (0, -1), (-1, 0), (-1, 0)]
    gr.add_edges(edges)
    g = gr.compressed(twoway=True, edge_nr_translation=True)
    g2 = pickle.loads(pickle.dumps(g))
    s = StringIO('')
    g.write_dot(s)
    s2 = StringIO('')
    g2.write_dot(s2)
    assert s.getvalue() == s2.getvalue()
    assert g.__getstate__() == g2.__getstate__()

    gr = Graph(allow_negative_nodes=False)
    edges = [(0, 1), (1, 2), (1, 2), (0, 2)]
    g = gr.compressed(twoway=False, edge_nr_translation=False)
    g2 = pickle.loads(pickle.dumps(g))
    s = StringIO('')
    g.write_dot(s)
    s2 = StringIO('')
    g2.write_dot(s2)
    assert s.getvalue() == s2.getvalue()
    assert g.__getstate__() == g2.__getstate__()
