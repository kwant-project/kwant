# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from defs cimport gint

cdef extern from "c_slicer/slicer.h":
    struct Slicing:
        int nslices
        int *slice_ptr
        int *slices

    Slicing *slice(gint, gint *, gint *, gint, gint *,
                   gint, gint *)

    void freeSlicing(Slicing *)
