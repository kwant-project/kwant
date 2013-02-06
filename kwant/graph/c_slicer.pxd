# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from defs cimport gint

cdef extern from "c_slicer/slicer.h":
   struct Slicing:
      int nslices
      int *slice_ptr, *slices

   Slicing *slice(gint, gint *, gint *, gint, gint *,
                  gint, gint *)

   void freeSlicing(Slicing *)
