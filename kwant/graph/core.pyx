# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Directed graphs optimized for storage and runtime efficiency."""

__all__ = ['Graph', 'CGraph']

# In this module we manage arrays directly with malloc, realloc and free to
# circumvent two problems with Cython:
#
# (1) There are no efficient arrays which allow appending (numpy.ndarray
#     doesn't).
#
# (2) Cython extension types cannot have typed buffers as members.
#
# Once these two problems are solved, the corresponding code could be
# rewritten, probably to use Python's array.array.

# TODO: represent all dangling nodes by -1

# TODO (perhaps): transform Graph into something which behaves like a python
# sequence.  Allow creation of compressed graphs from any sequence.

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
import numpy as np
cimport numpy as np
from kwant.graph.defs cimport gint

cdef class Graph:
    """An uncompressed graph.  Used to make compressed graphs.  (See `CGraph`.)
    """
    # The array edges holds `size` elements with space for `capacity`.

    def __init__(self, allow_negative_nodes=False):
        self.allow_negative_nodes = allow_negative_nodes

    def __dealloc__(self):
        free(self.edges)

    property num_nodes:
        def __get__(self):
            return self._num_nodes

        def __set__(self, value):
            if value < self._num_nodes:
                raise ValueError("The number of nodes cannot be decreased.")
            self._num_nodes = value

    cpdef reserve(self, gint capacity):
        """Reserve space for edges.

        Parameters
        ----------
        capacity : integer
           Number of edges for which to reserve space.

        Notes
        -----
        It is not necessary to call this method, but using it can speed up the
        creation of graphs.
        """
        if capacity <= self.capacity:
            return
        self.edges = <Edge*>realloc(self.edges, capacity * sizeof(Edge))
        if not self.edges:
            raise MemoryError
        self.capacity = capacity

    cpdef gint add_edge(self, gint tail, gint head) except -1:
        """Add the directed edge (`tail`, `head`) to the graph.

        Parameters
        ----------
        tail : integer
        head : integer

        Raises
        ------
        ValueError
            If a negative node is added when this has not been allowed
            explicitly or if an edge is doubly-dangling.

        Returns
        -------
        edge_nr : integer
           The sequential number of the edge.  This number can be used to query
           for the edge ID of an edge in the compressed graph.
        """
        cdef bint neg_tail = tail < 0
        cdef bint neg_head = head < 0
        if neg_tail or neg_head:
            if not self.allow_negative_nodes:
                raise ValueError(
                    "Negative node numbers have to be allowed explicitly.")
            if neg_tail and neg_head:
                raise ValueError("Doubly-dangling edges are never allowed.")
            if neg_head:
                self.num_pn_edges += 1
            else:
                self.num_np_edges += 1
        else:
            self.num_pp_edges += 1

        if self.size == self.capacity:
            if self.capacity == 0:
                self.reserve(8)
            else:
                self.reserve(2 * self.capacity)
        self.edges[self.size].tail = tail
        self.edges[self.size].head = head
        self.size += 1
        self._num_nodes = max(self._num_nodes, tail + 1, head + 1)
        return self.size - 1

    def add_edges(self, edges):
        """Add multiple edges in one pass.

        Parameters
        ----------
        edges : iterable of 2-sequences of integers
            The parameter `edges` must be an iterable of elements which
            describe the edges to be added.  For each edge-element, edge[0] and
            edge[1] must give, respectively, the tail and the head.  Valid
            edges are, for example, a list of 2-integer-tuples, or an
            numpy.ndarray of integers with a shape (n, 2).  The latter case is
            optimized.

        Returns
        -------
        first_edge_nr : integer
           The sequential number of the first of the added edges.  The numbers
           of the other added edges are consecutive integers following the
           number of the first.  Edge numbers can be used to query for the edge
           ID of an edge in the compressed graph.
        """
        result = self.size
        if isinstance(edges, np.ndarray):
            if edges.dtype == np.int64:
                self._add_edges_ndarray_int64(edges)
            elif edges.dtype == np.int32:
                self._add_edges_ndarray_int32(edges)
            else:
                self._add_edges_ndarray_int64(edges.astype(np.int64))
        else:
            for edge in edges:
                self.add_edge(*edge)
        return result

    cdef _add_edges_ndarray_int64(self, np.ndarray[np.int64_t, ndim=2] edges):
        cdef int i
        for i in range(edges.shape[0]):
            self.add_edge(edges[i, 0], edges[i, 1])

    cdef _add_edges_ndarray_int32(self, np.ndarray[np.int32_t, ndim=2] edges):
        cdef int i
        for i in range(edges.shape[0]):
            self.add_edge(edges[i, 0], edges[i, 1])

    def compressed(self, bint twoway=False, bint edge_nr_translation=False,
                   bint allow_lost_edges=False):
        """Build a CGraph from this graph.

        Parameters
        ----------
        twoway : boolean (default: False)
            If set, it will be possible to query the compressed graph for
            incoming neighbors.
        edge_nr_translation : boolean (default: False)
            If set, it will be possible to call the method `edge_id`.
        allow_lost_edges : boolean (default: False)
            If set, negative tails are accepted even with one-way compression.

        Raises
        ------
        ValueError
            When negative tails occur while `twoway` and `allow_lost_edges` are
            both false.

        Notes
        -----
        In a one-way compressed graph, an edge with a negative tail is present
        only minimally: it is only possible to query the head of such an edge,
        given the edge ID.  This is why one-way compression of a graph with a
        negative tail leads to a ValueError being raised, unless
        `allow_lost_edges` is true.
        """
        assert (self.size ==
                self.num_pp_edges + self.num_pn_edges + self.num_np_edges)
        if not (twoway or allow_lost_edges or self.num_np_edges == 0):
            raise ValueError('Edges with negative tails cannot be '
                             'represented in an one-way compressed graph.')

        cdef gint s, tail, head, edge_nr
        cdef CGraph_malloc result = CGraph_malloc(twoway, edge_nr_translation,
                                                  self._num_nodes,
                                                  self.num_pp_edges,
                                                  self.num_pn_edges,
                                                  self.num_np_edges)
        cdef gint *hbuf = result.heads_idxs + 1, *heads = result.heads
        cdef gint *tbuf = result.tails_idxs + 1, *tails = result.tails
        cdef gint *edge_ids = result.edge_ids
        cdef gint edge_id = 0, num_edges     # = 0 is there to silence warning.
        cdef Edge *edge

        # `hbuf` is just `heads_idxs` shifted by one.  We will use `hbuf` to
        # build up `heads` and its end state will be such that `heads_idxs`
        # will have the right content.  For `tbuf`, replace "head" with tail in
        # the previous text.

        # Make a histogram of outgoing edges per node in `hbuf` and one of
        # incoming edges per node in `tbuf`.
        memset(result.heads_idxs, 0, (self._num_nodes + 1) * sizeof(gint))
        if twoway:
            memset(result.tails_idxs, 0, (self._num_nodes + 1) * sizeof(gint))
        for edge in self.edges[:self.size]:
            if edge.tail >= 0:
                hbuf[edge.tail] += 1
            if twoway and edge.head >= 0:
                tbuf[edge.head] += 1

        # Replace `hbuf` with its "antiderivative" and then subtract the
        # original `hbuf` from it.  This is done in one pass.
        s = 0
        for tail in range(self._num_nodes):
            s += hbuf[tail]
            hbuf[tail] = s - hbuf[tail]

        # Same as before for `tbuf`.
        if twoway:
            s = 0
            for head in range(self._num_nodes):
                s += tbuf[head]
                tbuf[head] = s - tbuf[head]

        # Iterate through all edges and build `heads` and `tails`.
        next_np_edge_id = result.num_px_edges
        edge_nr = 0
        for edge in self.edges[:self.size]:
            if edge.tail >= 0:
                edge_id = hbuf[edge.tail]
                hbuf[edge.tail] += 1
            elif twoway:
                assert edge.head >= 0
                edge_id = next_np_edge_id
                next_np_edge_id += 1
            else:
                edge_id = -1
            if edge_id >= 0:
                heads[edge_id] = edge.head
            if twoway and edge.head >= 0:
                tails[tbuf[edge.head]] = edge.tail
                edge_ids[tbuf[edge.head]] = edge_id
                tbuf[edge.head] += 1
            if edge_nr_translation:
                result.edge_ids_by_edge_nr[edge_nr] = edge_id
                edge_nr += 1

        assert result.num_edges == next_np_edge_id
        return result

    def write_dot(self, file):
        """Write a representation of the graph in dot format to `file`.

        That resulting file can be visualized with dot(1) or neato(1) form the
        graphviz package.
        """
        cdef Edge *edge
        file.write("digraph g {\n")
        for edge in self.edges[:self.size]:
            file.write("  %d -> %d;\n" % (edge.tail, edge.head))
        file.write("}\n")


cdef class gintArraySlice:
    def __len__(self):
        return self.size

    def __getitem__(self, gint key):
        cdef gint index
        if key >= 0:
            index = key
        else:
            index = self.size + key
        if index < 0 or index >= self.size:
            raise IndexError('Index out of range.')
        return self.data[index]

cdef class EdgeIterator:
    def __iter__(self):
        return self

    def __next__(self):
        if self.edge_id == self.graph.num_edges:
            raise StopIteration
        cdef gint current_edge_id = self.edge_id
        self.edge_id += 1
        while current_edge_id >= self.graph.heads_idxs[self.tail + 1]:
            self.tail += 1
        return self.tail, self.graph.heads[current_edge_id]


class DisabledFeatureError(RuntimeError):
    pass


class NodeDoesNotExistError(IndexError):
    pass


class EdgeDoesNotExistError(IndexError):
    pass


_need_twoway = 'Enable "twoway" during graph compression.'

cdef class CGraph:
    """A compressed graph which can be efficiently queried for the existence of
    edges and outgoing neighbors.

    Objects of this class do not initialize the members themselves, but expect
    that they hold usable values.  A good way to create them is by compressing
    a `Graph`.

    Iterating over a graph yields a sequence of (tail, head) pairs of all
    edges.  The number of an edge in this sequence equals its edge ID.  The
    built-in function `enumerate` can thus be used to easily iterate over all
    edges along with their edge IDs.
    """
    def __iter__(self):
        """Return an iterator over (tail, head) of all edges."""
        cdef EdgeIterator result = EdgeIterator()
        result.graph = self
        result.edge_id = 0
        result.tail = 0
        return result

    def has_dangling_edges(self):
        return not self.num_edges == self.num_px_edges == self.num_xp_edges

    cpdef gintArraySlice out_neighbors(self, gint node):
        """Return the nodes a node points to.

        Parameters
        ----------
        node : integer

        Returns
        -------
        nodes : sequence of integers

        Raises
        ------
        NodeDoesNotExistError
        """
        if node < 0 or node >= self.num_nodes:
            raise NodeDoesNotExistError()
        cdef gintArraySlice result = gintArraySlice()
        result.data = &self.heads[self.heads_idxs[node]]
        result.size = &self.heads[self.heads_idxs[node + 1]] - result.data
        return result

    def out_edge_ids(self, gint node):
        """Return the IDs of outgoing edges of node.

        Parameters
        ----------
        node : integer

        Returns
        -------
        edge_ids : sequence of integers

        Raises
        ------
        NodeDoesNotExistError
        """
        if node < 0 or node >= self.num_nodes:
            raise NodeDoesNotExistError()
        return iter(xrange(self.heads_idxs[node], self.heads_idxs[node + 1]))

    def in_neighbors(self, gint node):
        """Return the nodes which point to a node.

        Parameters
        ----------
        node : integer

        Returns
        -------
        nodes : sequence of integers

        Raises
        ------
        NodeDoesNotExistError
        DisabledFeatureError
            If the graph is not two-way compressed.
        """
        if not self.twoway:
            raise DisabledFeatureError(_need_twoway)
        if node < 0 or node >= self.num_nodes:
            raise NodeDoesNotExistError()
        cdef gintArraySlice result = gintArraySlice()
        result.data = &self.tails[self.tails_idxs[node]]
        result.size = &self.tails[self.tails_idxs[node + 1]] - result.data
        return result

    def in_edge_ids(self, gint node):
        """Return the IDs of incoming edges of a node.

        Parameters
        ----------
        node : integer

        Returns
        -------
        edge_ids : sequence of integers

        Raises
        ------
        NodeDoesNotExistError
        DisabledFeatureError
            If the graph is not two-way compressed.
        """
        if not self.twoway:
            raise DisabledFeatureError(_need_twoway)
        if node < 0 or node >= self.num_nodes:
            raise NodeDoesNotExistError()
        cdef gintArraySlice result = gintArraySlice()
        result.data = &self.edge_ids[self.tails_idxs[node]]
        result.size = &self.edge_ids[self.tails_idxs[node + 1]] - result.data
        return result

    def has_edge(self, gint tail, gint head):
        """Does the graph contain the edge (tail, head)?

        Parameters
        ----------
        tail : integer
        head : integer

        Returns
        -------
        had_edge : boolean

        Raises
        ------
        NodeDoesNotExistError
        EdgeDoesNotExistError
        DisabledFeatureError
            If `tail` is negative and the graph is not two-way compressed.
        """
        cdef gint h, t
        if tail >= self.num_nodes or head >= self.num_nodes:
            raise NodeDoesNotExistError()
        if tail >= 0:
            for h in self.heads[self.heads_idxs[tail]
                                : self.heads_idxs[tail + 1]]:
                if h == head: return True
        else:
            if not self.twoway:
                raise DisabledFeatureError(_need_twoway)
            if head < 0:
                raise EdgeDoesNotExistError()
            for t in self.tails[self.tails_idxs[head]
                                : self.tails_idxs[head + 1]]:
                if t == tail: return True
        return False

    def edge_id(self, gint edge_nr):
        """Return the edge ID of an edge given its sequential number.

        Parameters
        ----------
        edge_nr : integer

        Returns
        -------
        edge_id : integer

        Raises
        ------
        DisabledFeatureError
            If `edge_nr_translation` was not enabled during graph compression.
        EdgeDoesNotExistError
        """
        if not self.edge_ids_by_edge_nr:
            raise DisabledFeatureError(
                'Enable "edge_nr_translation" during graph compression.')
        if edge_nr < 0 or edge_nr >= self.edge_nr_end:
            raise EdgeDoesNotExistError()
        result = self.edge_ids_by_edge_nr[edge_nr]
        if result < 0:
            raise EdgeDoesNotExistError()
        return result

    def first_edge_id(self, gint tail, gint head):
        """Return the edge ID of the first edge (tail, head).

        Parameters
        ----------
        tail : integer
        head : integer

        Returns
        -------
        edge_id : integer

        Raises
        ------
        NodeDoesNotExist
        EdgeDoesNotExistError
        DisabledFeatureError
            If `tail` is negative and the graph is not two-way compressed.

        Notes
        -----
        This method is useful for graphs where each edge occurs only once.
        """
        if tail >= self.num_nodes or head >= self.num_nodes:
            raise NodeDoesNotExistError()
        if tail >= 0:
            for head_index in xrange(self.heads_idxs[tail],
                                     self.heads_idxs[tail + 1]):
                if self.heads[head_index] == head:
                    return head_index
        else:
            if not self.twoway:
                raise DisabledFeatureError(_need_twoway)
            for tail_index in xrange(self.tails_idxs[head],
                                     self.tails_idxs[head + 1]):
                if self.tails[tail_index] == tail:
                    return self.edge_ids[tail_index]
        raise EdgeDoesNotExistError()

    def all_edge_ids(self, gint tail, gint head):
        """Return an iterator over all edge IDs of edges with a given tail and
        head.

        Parameters
        ----------
        tail : integer
        head : integer

        Returns
        -------
        edge_id : integer

        Raises
        ------
        NodeDoesNotExist
        EdgeDoesNotExistError
        DisabledFeatureError
            If `tail` is negative and the graph is not two-way compressed.
        """
        if tail >= self.num_nodes or head >= self.num_nodes:
            raise NodeDoesNotExistError()
        result = []
        if tail >= 0:
            for head_index in xrange(self.heads_idxs[tail],
                                     self.heads_idxs[tail + 1]):
                if self.heads[head_index] == head:
                    result.append(head_index)
        else:
            if not self.twoway:
                raise DisabledFeatureError(_need_twoway)
            for tail_index in xrange(self.tails_idxs[head],
                                     self.tails_idxs[head + 1]):
                if self.tails[tail_index] == tail:
                    result.append(self.edge_ids[tail_index])
        return result

    # TODO: optimize this for the case of twofold graphs and low degree.
    def tail(self, gint edge_id):
        """Return the tail of an edge, given its edge ID.

        Parameters
        ----------
        edge_id : integer

        Returns
        -------
        tail : integer
            If the edge exists and is positive.
        None
            If the tail is negative.

        Raises
        ------
        EdgeDoesNotExistError

        Notes
        -----
        The average performance of this method is O(log num_nodes) for
        non-negative tails and O(1) for negative ones.
        """
        if edge_id < 0 or edge_id >= self.num_edges:
            raise EdgeDoesNotExistError
        if edge_id >= self.num_px_edges:
            assert self.twoway
            return None
        cdef gint lower = 0, upper = self.num_nodes, tail = 0
        while upper - lower > 1:
            tail = (upper + lower) // 2
            if edge_id == self.heads_idxs[tail]:
                return tail
            if edge_id < self.heads_idxs[tail]:
                upper = tail
            else:
                lower = tail
        return lower

    def head(self, gint edge_id):
        """Return the head of an edge, given its edge ID.

        Parameters
        ----------
        edge_id : integer

        Raises
        ------
        EdgeDoesNotExistError

        Notes
        -----
        This method executes in constant time.  It works for all edge IDs,
        returning both positive and negative heads.
        """
        if edge_id < 0 or edge_id >= self.num_edges:
            raise EdgeDoesNotExistError()
        return self.heads[edge_id]

    def write_dot(self, file):
        """Write a representation of the graph in dot format to `file`.

        Parameters
        ----------
        file : file-like object

        Notes
        -----
        That resulting file can be visualized with dot(1) or neato(1) form the
        `graphviz <http://graphviz.org/>`_ package.
        """
        cdef gint tail
        file.write("digraph g {\n")
        for tail in range(self.num_nodes):
            for head in self.heads[self.heads_idxs[tail]
                                   : self.heads_idxs[tail + 1]]:
                file.write("  %d -> %d;\n" % (tail, head))
        file.write("}\n")


cdef class CGraph_malloc(CGraph):
    """A CGraph which allocates and frees its own memory."""

    def __cinit__(self, twoway, edge_nr_translation, num_nodes,
                  num_pp_edges, num_pn_edges, num_np_edges):
        self.twoway = twoway
        self.edge_nr_translation = edge_nr_translation
        self.num_nodes = num_nodes
        self.num_px_edges = num_pp_edges + num_pn_edges
        self.edge_nr_end = num_pp_edges + num_pn_edges + num_np_edges

        self.heads_idxs = <gint*>malloc((num_nodes + 1) * sizeof(gint))
        if self.twoway:
            # The graph is two-way. n->p edges will exist in the compressed
            # graph.
            self.num_xp_edges = num_pp_edges + num_np_edges
            self.num_edges = self.edge_nr_end
            self.tails_idxs = <gint*>malloc((num_nodes + 1) * sizeof(gint))
            self.tails = <gint*>malloc(
                self.num_xp_edges * sizeof(gint))
            self.edge_ids = <gint*>malloc(
                self.num_xp_edges * sizeof(gint))
        else:
            # The graph is one-way. n->p edges will be ignored.
            self.num_xp_edges = num_pp_edges
            self.num_edges = self.num_px_edges
        self.heads = <gint*>malloc(self.num_edges * sizeof(gint))
        if edge_nr_translation:
            self.edge_ids_by_edge_nr = <gint*>malloc(
                self.edge_nr_end * sizeof(gint))
        if (not self.heads_idxs or not self.heads
            or (twoway and (not self.tails_idxs
                             or not self.tails
                             or not self.edge_ids))
            or (edge_nr_translation and not self.edge_ids_by_edge_nr)):
            raise MemoryError

    def __dealloc__(self):
        free(self.edge_ids_by_edge_nr)
        free(self.heads)
        free(self.edge_ids)
        free(self.tails)
        free(self.tails_idxs)
        free(self.heads_idxs)
