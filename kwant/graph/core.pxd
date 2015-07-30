# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

cimport numpy as np
from .defs cimport gint

cdef struct Edge:
    gint tail, head

cdef class Graph:
    cdef int allow_negative_nodes
    cdef Edge *edges
    cdef gint capacity, size, _num_nodes
    cdef gint num_pp_edges, num_pn_edges, num_np_edges

    cpdef reserve(self, gint capacity)
    cpdef gint add_edge(self, gint tail, gint head) except -1
    cdef _add_edges_ndarray_int64(self, np.ndarray[np.int64_t, ndim=2] edges)
    cdef _add_edges_ndarray_int32(self, np.ndarray[np.int32_t, ndim=2] edges)

cdef class gintArraySlice:
    cdef gint *data
    cdef gint size

cdef class CGraph:
    cdef readonly bint twoway, edge_nr_translation
    cdef readonly gint num_nodes, num_edges, num_px_edges, num_xp_edges
    cdef gint *heads_idxs
    cdef gint *heads
    cdef gint *tails_idxs
    cdef gint *tails
    cdef gint *edge_ids
    cdef gint *edge_ids_by_edge_nr
    cdef gint edge_nr_end

    cpdef gintArraySlice out_neighbors(self, gint node)


cdef class CGraph_malloc(CGraph):
    pass

cdef class EdgeIterator:
    cdef CGraph graph
    cdef gint edge_id, tail
