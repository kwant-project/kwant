from defs cimport gint

cdef extern from "c_slicer/slicer.h":
   struct Slicing:
      int nslices
      int *slice_ptr, *slices

   Slicing *slice(gint, gint *, gint *, gint, gint *,
                  gint, gint *)

   void freeSlicing(Slicing *)
