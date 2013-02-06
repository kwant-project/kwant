// Copyright 2011-2013 kwant authors.
//
// This file is part of kwant.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of kwant authors can be found in
// the AUTHORS file at the top-level directory of this distribution and at
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
