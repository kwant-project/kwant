# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Utilities to modify compressed graphs"""

__all__ = ['make_undirected', 'remove_duplicates', 'induced_subgraph',
           'print_graph']

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
cimport cython
from kwant.graph.defs cimport gint
from .defs import gint_dtype
from kwant.graph.core cimport CGraph, CGraph_malloc
from .core import CGraph, CGraph_malloc

@cython.boundscheck(False)
def make_undirected(CGraph gr, remove_dups=True, calc_weights=False):
    """undirected_graph(gr) expects a CGraph gr as input, which is interpreted
    as a directed graph, and returns a CGraph that is explicitely undirected,
    i.e. for every edge (i,j) there is also the edge (j,i). In the process, the
    function also removes all 'dangling' links, i.e. edges to or from
    negative node numbers.

    If remove_dups == True (default value is True), any duplicates of edges
    will be removed (this applies to the case where there are multiple edges
    (i,j), not to having (i,j) and (j,i)).

    The effect of the duplicate edges can be retained if calc_weights == True
    (default value is False), in which case a weight array is returned
    containing the multiplicity of the edges after the graph has been made
    undirected.

    As a (somewhat drastic but illustrative) example, if make_undirected is
    applied to a undirected graph, it will return the same graph again
    (possibly with the order of edges changed) and a weight array with 2
    everywhere.  (Of course, in this case one does not need to call
    make_undirected ...)

    make_undirected() will always return a one-way graph, regardless of
    whether the input was a two-way graph or not (NOTE: This
    restriction could be lifted, if necessary). In addition, the
    original edge_ids are lost -- the resulting graph will have
    edge_ids that are not related to the original ones. (NOTE: there
    certainly is a relation, but as long as no-one needs it it remains
    unspecified)
    """

    cdef gint i, j, p

    # The undirected graph will have twice as many edges than the directed one
    # (duplicates will be deleted afterwards).
    cdef CGraph_malloc ret = CGraph_malloc(False, False, gr.num_nodes,
                                           gr.heads_idxs[gr.num_nodes] * 2,
                                           0, 0)

    # In the following we build up the Graph directly in compressed format by
    # adding for every edge (i,j) [with i,j>=0] also the edge (j,i). Taking
    # care of possible doubling is done in a second step later.

    # Initialize the new index array:
    # First, compute a histogram of edges.
    memset(ret.heads_idxs, 0, (ret.num_nodes + 1) * sizeof(gint))

    # This is using Christoph's trick of building up the graph without
    # additional buffer array.
    cdef gint *buffer = ret.heads_idxs + 1

    for i in xrange(gr.num_nodes):
        for p in xrange(gr.heads_idxs[i], gr.heads_idxs[i+1]):
            if gr.heads[p] >= 0:
                buffer[i] += 1
                buffer[gr.heads[p]] += 1

    cdef gint s = 0
    for i in xrange(ret.num_nodes):
        s += buffer[i]
        buffer[i] = s - buffer[i]

    for i in xrange(gr.num_nodes):
        for p in xrange(gr.heads_idxs[i], gr.heads_idxs[i+1]):
            j = gr.heads[p]
            if j >= 0:
                ret.heads[buffer[i]] = j
                buffer[i] += 1
                ret.heads[buffer[j]] = i
                buffer[j] += 1

    ret.num_edges = ret.heads_idxs[ret.num_nodes]

    # Now remove duplicates if desired.
    cdef np.ndarray[gint, ndim=1] weights

    if calc_weights:
        weights = np.empty(ret.heads_idxs[ret.num_nodes], dtype=gint_dtype)
        weights[:] = 1

    if remove_dups:
        if calc_weights:
            remove_duplicates(ret, weights)
        else:
            remove_duplicates(ret)

    if calc_weights:
        return ret, weights
    else:
        return ret


@cython.boundscheck(False)
def remove_duplicates(CGraph gr, np.ndarray[gint, ndim=1] edge_weights=None):
    """Remove duplicate edges in the CGraph gr (this applies to the case where
    there are multiple edges (i,j), not to having (i,j) and (j,i)). This
    function modifes the graph in place.

    If edge_weights is provided, edge_weights is modified such that the new
    edge weights are the sum of the old edge weights if there are duplicate
    edges.

    This function only works on simple graphs (not two-way graphs), and
    it does not work on graphs which have a relation between the edge number
    (given by the order the edges are added) and the edge_id (given by the
    order the edges appear in the graph), see the documentation of CGraph.
    (Both restrictions could be lifted if necessary.) Furthermore, the
    function does not support negative node numbers, i.e. dangling links
    (the concept of being duplicate is more complicated there.)
    """
    cdef gint i, j , p, q, nnz
    cdef np.ndarray[gint, ndim=1] w

    if gr.twoway:
        raise RuntimeError("remove_duplicates does not support two-way "
                           "graphs")

    if gr.edge_ids_by_edge_nr:
        raise RuntimeError("remove_duplicates does not support graphs with "
                           "a relation between the edge number and the edge "
                           "id")

    # The array w will hold the position of head j in the heads array.
    w = np.empty(gr.num_nodes, dtype=gint_dtype)
    w[:]=-1

    nnz=0

    for i in xrange(gr.num_nodes):
        q = nnz

        for p in xrange(gr.heads_idxs[i], gr.heads_idxs[i+1]):
            j = gr.heads[p]

            # Check if we have found a previous entry (i,j).  (In this case w
            # will have an index larger than the indices of all previous edges
            # with tails < i, as stored in q.)
            if w[j] >= q:
                # entry is a duplicate
                if edge_weights != None:
                    edge_weights[w[j]] += edge_weights[p]
            else:
                w[j] = nnz
                gr.heads[nnz] = j
                nnz += 1

        # Fix the index array.
        gr.heads_idxs[i] = q

    # Finally the new number of nonzeros
    gr.heads_idxs[gr.num_nodes] = nnz
    gr.num_edges = nnz

    # Release memory that is not needed any more.
    gr.heads = <gint *>realloc(gr.heads, nnz * sizeof(gint))
    if not gr.heads:
        raise MemoryError

    if edge_weights != None:
        edge_weights.resize(nnz, refcheck=False)

@cython.boundscheck(False)
def induced_subgraph(CGraph gr, select,
                     np.ndarray[gint, ndim=1] edge_weights=None):
    """Return a subgraph of the CGraph gr by picking all nodes
    [0:gr.num_nodes] for which select is True. select can be either a
    NumPy array, or a function that takes the node number as
    input. This function returns a CGraph as well.

    The nodes in the new graph are again numbered sequentially from 0
    to num_nodes-1, where num_nodes is the number of nodes in the
    subgraph. The numbering is done such that the ordering of the node
    numbers in the original and the subgraph are preserved (i.e.
    if nodes n1 and n2 are both in the subgraph, and
    original node number of n1 < original node number of n2,
    then also subgraph node number n1 < subgraph node number n2).

    If edge_weights is provided, the function also returns the edge
    weights for the subgraph which are simply a subset of the original
    weights.

    This function returns a simple graph, regardless of whether the
    input was a two-way graph or not (NOTE: This restriction could be
    lifted, if necessary). Also, the resulting edge_ids are not
    related to the original ones in any way (NOTE: There certainly is
    a relation, but as long no-one needs it, we do not specify
    it). Also, negative nodes are discarded (NOTE: this restriction
    can also be lifted).
    """

    cdef np.ndarray[gint, ndim=1] indextab
    cdef CGraph_malloc subgr
    cdef np.ndarray[gint, ndim=1] sub_edge_weights
    cdef gint sub_num_nodes, sub_num_edges
    cdef gint i, iedge, edge_count

    # First figure out the new number of nodes.
    sub_num_nodes = 0
    indextab = np.empty(gr.num_nodes, dtype=gint_dtype)
    indextab[:] = -1

    # Pre-evaluating the select functions seems to be more than twice as fast
    # as calling select() repeatedly in the loop.  The thing is that one cannot
    # type ndarray as bool in Cython (yet) [Taking bint results in a strange
    # crash].  It would be possible to cast it into a cython type using
    # .astype(), but this didn't seem to make any relevant speed difference.
    if isinstance(select, np.ndarray):
        selecttab = select
    else:
        selecttab = select(np.arange(gr.num_nodes, dtype=gint_dtype))

    for i in xrange(gr.num_nodes):
        if selecttab[i]:
            indextab[i] = sub_num_nodes
            sub_num_nodes += 1

    # Now count the number of new edges.
    sub_num_edges = 0

    for i in xrange(gr.num_nodes):
        if indextab[i] > -1:
            for iedge in xrange(gr.heads_idxs[i], gr.heads_idxs[i + 1]):
                if indextab[gr.heads[iedge]] > -1:
                    sub_num_edges += 1

    # Allocate the new graph.
    subgr = CGraph_malloc(False, False, sub_num_nodes, sub_num_edges, 0, 0)

    if edge_weights != None:
        sub_edge_weights = np.empty(sub_num_edges, dtype=gint_dtype)

    # Now fill the new edge array.
    edge_count = 0

    for i in xrange(gr.num_nodes):
        if indextab[i]>-1:
            subgr.heads_idxs[indextab[i]] = edge_count
            for iedge in xrange(gr.heads_idxs[i], gr.heads_idxs[i+1]):
                if indextab[gr.heads[iedge]] > -1:
                    subgr.heads[edge_count] = indextab[gr.heads[iedge]]
                    if edge_weights != None:
                        sub_edge_weights[edge_count] = edge_weights[iedge]
                    edge_count += 1
    subgr.heads_idxs[sub_num_nodes] = edge_count

    subgr.num_edges = edge_count

    if edge_weights != None:
        return subgr, sub_edge_weights
    else:
        return subgr


def print_graph(gr):
    for i in xrange(gr.num_nodes):
        print i," -> ",
        for j in gr.out_neighbors(i):
            print j,
        print
