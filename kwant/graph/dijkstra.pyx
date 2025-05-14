#
# (2018) Modified by Kwant Authors
#
# Modifications
# =============
# Merged and modified from scipy/sparse/csgraph/_shortest_path.pyx
#
# All shortest path algorithms except for Dijkstra's removed.
# Implementation of Dijkstra's algorithm modified to allow for specific
# use-cases required by Flux. The changes are documented in the docstring.
#
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2017 SciPy Developers.
# All rights reserved.
#
# Copyright (c) 2011 Jake Vanderplas <vanderplas@astro.washington.edu>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of Enthought nor the names of the SciPy Developers
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY as inf

from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t
ITYPE = np.int32


def dijkstra_directed(
        object graph,
        ITYPE_t[:] sources,
        ITYPE_t[:] targets,
        bint return_paths=True,
        bint return_predecessors=False):
    """Modified directed Dijkstra algorithm.

    Edges with infinite weight are treated as if they do not exist.

    The shortest paths between edge pairs 'zip(sources, targets)'
    are found.

    If 'len(sources) == 1' then this routine can be used to
    compute the one-to-all distances by passing an integer
    greater than any node in 'graph' as the 'target'.
    In this case 'return_predecessors' may be specified, and
    the predecessor matrix will also be returned.

    'return_predecessors' and 'return_paths' are mutually exclusive.

    Returns
    -------
    if return_paths:
        (paths, path_lengths)
    elif return_predecessors:
        (path_lengths, predecessors)
    else:
        path_lengths

    """
# Implementation of Dijkstra's algorithm modified to allow for
# early exit (when target is found) and return the path from source
# to target, rather than the whole predecessor matrix. In addition
# graph edges with infinite weight are treated as if they do not exist.

    cdef ITYPE_t[:] csr_indices = graph.indices, csr_indptr = graph.indptr
    cdef DTYPE_t[:] csr_weights = graph.data

    # This implementation of Dijkstra's algorithm is very tightly coupled
    # to our use-case in 'flux', so we allow ourselves to assert
    assert sources.shape[0] == targets.shape[0]
    assert graph.shape[0] == graph.shape[1]
    assert not (return_predecessors and return_paths)
    assert not (return_predecessors and (sources.shape[0] > 1))

    cdef unsigned int num_links = sources.shape[0], num_nodes = graph.shape[0]

    cdef unsigned int i, k, j_source, j_target, j_current
    cdef ITYPE_t j

    cdef DTYPE_t next_val

    cdef FibonacciHeap heap
    cdef FibonacciNode *v
    cdef FibonacciNode *current_node
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(num_nodes * sizeof(FibonacciNode))

    cdef ITYPE_t[:] pred = np.empty((num_nodes,), dtype=ITYPE)


    # outputs
    cdef DTYPE_t[:] path_lengths = np.zeros((num_links,), float)
    cdef list paths
    if return_paths:
        paths = []

    for i in range(num_links):
        j_source = sources[i]
        j_target = targets[i]

        for k in range(num_nodes):
            initialize_node(&nodes[k], k)
        pred[:] = -1  # only useful for debugging

        heap.min_node = NULL
        insert_node(&heap, &nodes[j_source])

        while heap.min_node:
            v = remove_min(&heap)
            v.state = SCANNED

            if v.index == j_target:
                path_lengths[i] = v.val
                if return_paths:
                    paths.append(_calculate_path(pred, j_source, j_target))
                break  # next iteration of outer 'for' loop

            for j in range(csr_indptr[v.index], csr_indptr[v.index + 1]):
                if csr_weights[j] == inf:
                    # Treat infinite weight links as missing
                    continue
                j_current = csr_indices[j]
                current_node = &nodes[j_current]
                if current_node.state != SCANNED:
                    next_val = v.val + csr_weights[j]
                    if current_node.state == NOT_IN_HEAP:
                        current_node.state = IN_HEAP
                        current_node.val = next_val
                        insert_node(&heap, current_node)
                        pred[j_current] = v.index
                    elif current_node.val > next_val:
                        decrease_val(&heap, current_node,
                                     next_val)
                        pred[j_current] = v.index

    free(nodes)
    if return_paths:
      return paths, path_lengths
    elif return_predecessors:
      return path_lengths, pred
    else:
        return path_lengths


cdef list _calculate_path(ITYPE_t[:] pred, int j_source, int j_target):
    visited = []
    cdef int node = j_target
    while node != j_source:
        visited.append(node)
        node = pred[node]
    visited.append(j_source)
    return visited


######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#
cdef enum FibonacciState:
    SCANNED
    NOT_IN_HEAP
    IN_HEAP


cdef struct FibonacciNode:
    unsigned int index
    unsigned int rank
    FibonacciState state
    DTYPE_t val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children


cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0):
    # Assumptions: - node is a valid pointer
    #              - node is not currently part of a heap
    node.index = index
    node.val = val
    node.rank = 0
    node.state = NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL


cdef FibonacciNode* rightmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.right_sibling):
        temp = temp.right_sibling
    return temp


cdef FibonacciNode* leftmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp


cdef void add_child(FibonacciNode* node, FibonacciNode* new_child):
    # Assumptions: - node is a valid pointer
    #              - new_child is a valid pointer
    #              - new_child is not the sibling or child of another node
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:
        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1


cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling):
    # Assumptions: - node is a valid pointer
    #              - new_sibling is a valid pointer
    #              - new_sibling is not the child or sibling of another node
    cdef FibonacciNode* temp = rightmost_sibling(node)
    temp.right_sibling = new_sibling
    new_sibling.left_sibling = temp
    new_sibling.right_sibling = NULL
    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1


cdef void remove(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    if node.parent:
        node.parent.rank -= 1
        if node.left_sibling:
            node.parent.children = node.left_sibling
        elif node.right_sibling:
            node.parent.children = node.right_sibling
        else:
            node.parent.children = NULL

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.


cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    if heap.min_node:
        add_sibling(heap.min_node, node)
        if node.val < heap.min_node.val:
            heap.min_node = node
    else:
        heap.min_node = node


cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval):
    # Assumptions: - heap is a valid pointer
    #              - newval <= node.val
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    #              - node is in the heap
    node.val = newval
    if node.parent and (node.parent.val >= newval):
        remove(node)
        insert_node(heap, node)
    elif heap.min_node.val > node.val:
        heap.min_node = node


cdef void link(FibonacciHeap* heap, FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is already within heap

    cdef FibonacciNode *linknode
    cdef FibonacciNode *parent
    cdef FibonacciNode *child

    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)


cdef FibonacciNode* remove_min(FibonacciHeap* heap):
    # Assumptions: - heap is a valid pointer
    #              - heap.min_node is a valid pointer
    cdef FibonacciNode *temp
    cdef FibonacciNode *temp_right
    cdef FibonacciNode *out
    cdef unsigned int i

    # make all min_node children into root nodes
    if heap.min_node.children:
        temp = leftmost_sibling(heap.min_node.children)
        temp_right = NULL

        while temp:
            temp_right = temp.right_sibling
            remove(temp)
            add_sibling(heap.min_node, temp)
            temp = temp_right

        heap.min_node.children = NULL

    # choose a root node other than min_node
    temp = leftmost_sibling(heap.min_node)
    if temp == heap.min_node:
        if heap.min_node.right_sibling:
            temp = heap.min_node.right_sibling
        else:
            out = heap.min_node
            heap.min_node = NULL
            return out

    # remove min_node, and point heap to the new min
    out = heap.min_node
    remove(heap.min_node)
    heap.min_node = temp

    # re-link the heap
    for i in range(100):
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    return out
