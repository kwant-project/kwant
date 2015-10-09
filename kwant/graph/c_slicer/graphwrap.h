// Copyright 2011-2013 Kwant authors.
//
// This file is part of Kwant.  It is subject to the license terms in the file
// LICENSE.rst found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of Kwant authors can be found in
// the file AUTHORS.rst at the top-level directory of this distribution and at
// http://kwant-project.org/authors.

//-*-C++-*-
#ifndef _GRAPH_WRAPPER_H
#define _GRAPH_WRAPPER_H

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>

class GraphWrapper
{
public:
  //Some helper classes

  //functor to compare the degree of vertices
  class DegreeComparator
  {
  private:
    const GraphWrapper &graph;

  public:
    DegreeComparator( const GraphWrapper &_graph ) : graph(_graph)
    {
    }

    bool operator()( int _vertex1, int _vertex2 )
    {
      return graph.getEdges(_vertex1).size() < graph.getEdges(_vertex2).size();
    }
  };

  template<typename T>
  class VectorProxy {
    T *begin_it, *end_it;

  public:
    VectorProxy(T *_begin_it, T *_end_it) :
      begin_it(_begin_it), end_it(_end_it)
    {}

    T *begin() const
    {
      return begin_it;
    }

    T *end() const
    {
      return end_it;
    }

    size_t size() const
    {
      return end_it-begin_it;
    }
  };


protected:
  //data structure to hold graph in compressed form
  int *vertex_ptr;
  int *edges;

  int vertex_num;

public:
  GraphWrapper(int _vnum, int *_vertex_ptr, int *_edges) :
    vertex_ptr(_vertex_ptr), edges(_edges), vertex_num(_vnum)
  {
  }

public:
   //information about the graph

   //! number of vertices in the graph
   inline int size () const
   {
     return vertex_num;
   }
   //!Get the total number of edges in the graph
   inline int edgeSize() const
   {
     return vertex_ptr[vertex_num];
   }

  //! functions for accessing the edge structure
  inline VectorProxy<int> getEdges( int _vertex ) const
  {
    return VectorProxy<int>(edges+vertex_ptr[_vertex],
			    edges+vertex_ptr[_vertex+1]);
  }
};

#endif
