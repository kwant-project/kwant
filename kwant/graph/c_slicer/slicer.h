// Copyright 2011-2013 Kwant authors.
//
// This file is part of Kwant.  It is subject to the license terms in the file
// LICENSE.rst found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of Kwant authors can be found in
// the file AUTHORS.rst at the top-level directory of this distribution and at
// http://kwant-project.org/authors.

struct Slicing
{
  int nslices;
  int *slice_ptr, *slices;
};

#ifdef __cplusplus
extern "C"
#endif
struct Slicing *slice(int, int *, int *, int, int *,
                      int, int *);

#ifdef __cplusplus
extern "C"
#endif
void freeSlicing(struct Slicing *);
