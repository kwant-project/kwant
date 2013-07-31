// Copyright 2011-2013 Kwant authors.
//
// This file is part of Kwant.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of Kwant authors can be found in
// the AUTHORS file at the top-level directory of this distribution and at
// http://kwant-project.org/authors.

#include <algorithm>
#include <exception>

#include "graphwrap.h"
#include "partitioner.h"

#include "slicer.h"

extern "C"
Slicing *slice(int _node_num,
               int *_vertex_ptr,
               int *_edges,
               int _left_num, int *_left,
               int _right_num, int *_right)
{
  GraphWrapper graph(_node_num, _vertex_ptr,
                     _edges);

  vector<int> left(_left, _left+_left_num),
    right(_right, _right+_right_num);

  Partitioner parts(graph, left, right);

  Slicing *slicing;

  try {
    slicing=new Slicing;

    slicing->nslices=parts.parts.size();
    slicing->slice_ptr=new int[parts.parts.size()+1];
    slicing->slices=new int[graph.size()];
  }
  catch(std::bad_alloc &ba) {
    return NULL;
  }

  slicing->slice_ptr[0]=0;
  for(size_t i=0; i<parts.parts.size(); i++) {
    std::copy(parts.parts[i].begin(),
              parts.parts[i].end(),
              slicing->slices+slicing->slice_ptr[i]);
    slicing->slice_ptr[i+1]=slicing->slice_ptr[i]+
      parts.parts[i].size();
  }

  return slicing;
}

extern "C"
void freeSlicing(Slicing *_slicing)
{
  if(_slicing) {
    delete [] _slicing->slices;
    delete [] _slicing->slice_ptr;
    delete _slicing;
  }
}
