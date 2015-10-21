# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
cimport numpy as np
cimport cython
from libc.string cimport memcpy
from .defs cimport gint
from .defs import gint_dtype
from .core cimport CGraph

cimport c_slicer
# TODO: Once cythonize() no longer warns about it, replace the above be
# from . cimport c_slicer

__all__ = ['slice']

@cython.boundscheck(False)
def slice(CGraph graph, left, right):
    """
    TODO: write me.
    """
    cdef np.ndarray[gint, ndim=1] leftarr, rightarr, slc
    cdef c_slicer.Slicing *slicing
    cdef int i, slc_size

    leftarr = np.unique(np.array(left, dtype=gint_dtype))
    rightarr = np.unique(np.array(right, dtype=gint_dtype))

    if leftarr.ndim != 1:
        raise ValueError("Left cannot be interpreted as a 1D array.")

    if rightarr.ndim != 1:
        raise ValueError("Right cannot be interpreted as a 1D array.")

    if leftarr.size == 0 or rightarr.size == 0:
        raise ValueError("Empty boundary arrays are not supported yet.")

    # slicing only possible if there is no overlap between
    # left and right slices
    if np.intersect1d(rightarr, leftarr, assume_unique=True).size:
        return [tuple(xrange(graph.num_nodes))]

    slicing = c_slicer.slice(graph.num_nodes,
                             graph.heads_idxs,
                             graph.heads,
                             leftarr.size, <gint *>leftarr.data,
                             rightarr.size, <gint *>rightarr.data)
    slices = []
    for i in xrange(slicing.nslices):
        slc_size = slicing.slice_ptr[i+1] - slicing.slice_ptr[i]
        slc = np.empty(slc_size, dtype=gint_dtype)
        memcpy(slc.data, slicing.slices + slicing.slice_ptr[i],
               sizeof(gint) * slc_size)
        slices.append(slc)
    c_slicer.freeSlicing(slicing)
    return slices
