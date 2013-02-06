// Copyright 2011-2013 kwant authors.
//
// This file is part of kwant.  It is subject to the license terms in the
// LICENSE file found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of kwant authors can be found in
// the AUTHORS file at the top-level directory of this distribution and at
// http://kwant-project.org/authors.

//-*-C++-*-
#ifndef _BLOCK_TRIDIAGONAL_PARTITIONER_H
#define _BLOCK_TRIDIAGONAL_PARTITIONER_H

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>

#include "graphwrap.h"
#include "bucket_list.h"
//#include "graphalgorithms.h"

/** @short the first graph-partitioner
 ** *******************************************/
class Partitioner
{
public:
  enum Distributors {
    Distribute_Natural_Unbalanced,
    Distribute_Natural_Balanced,
    Distribute_Randomly };

  enum Optimizers {
    Optimize_No,
    Optimize_FM_MinCut,
    Optimize_FM_MinNetCut,
    Optimize_FM_MinNetCutMinCut } ;

public:
  const GraphWrapper &graph;

  vector<vector<int> > parts;

  vector<int> inSlice;
  vector<int> sliceIndex;

  enum Distributors distributor;
  enum Optimizers optimizer;

public:
  /** @param _graph           the underlying grid
   ** @param _tolerance
   ** @param _min_opt_passes
   ** @param _max_opt_passes
   ** @param _distributor
   ** @param _optimizer
   ** @param _mode            default is TwoBlocks, OneBlock is not implemented yet
   ** Default behaviour: TwoBlocks, left lead = 1, right lead = 2
   ** **************************************************************/
  template<typename Grid>
  Partitioner( const Grid  &_graph,
	       std::vector<int> &_left,
	       std::vector<int> &_right,
	       double       _tolerance = 0.01,
	       int          _min_opt_passes = 10,
	       int          _max_opt_passes = 10,
	       Distributors _distributor = Distribute_Natural_Balanced,
	       Optimizers   _optimizer   = Optimize_FM_MinNetCutMinCut) :
    graph(_graph), parts(1, vector<int>(0)),
    inSlice(_graph.size(),-1), sliceIndex(_graph.size(),0),
    distributor(_distributor), optimizer(_optimizer)
  {
    parts[0].reserve(_graph.size());

    for(int i=0; i<graph.size(); i++) {
      parts[0].push_back(i);

      inSlice[i]=0;
      sliceIndex[i]=parts[0].size()-1;
    }

    bisectFirst(_left, _right, _tolerance, _min_opt_passes, _max_opt_passes);
  }

private:
  /** @short first bisection? **/
  void bisectFirst( std::vector<int> &, std::vector<int> &, double _tolerance,
		    int _min_opt_passes, int _max_opt_passes);

  void bisect(int _part, int _part_slices,
	      std::vector<int> &, std::vector<int> &,
	      int _max_edges, double _tolerance,
	      int _min_opt_passes, int _max_opt_passes);

  //interface function that chooses between the different Optimizers
  inline int optimize(int _left_index, int _left_size,
		      int _right_index, int _right_size,
		      std::vector<bool> &_locked, int _pmax, int _tolerance)
  {
    if(optimizer == Optimize_FM_MinCut) {
      return FM_MinCut_optimize(_left_index, _left_size,
				_right_index, _right_size,
				_locked, _pmax, _tolerance);
    }
    else if(optimizer == Optimize_FM_MinNetCut) {
      return FM_MinNetCut_optimize(_left_index, _left_size,
				   _right_index, _right_size,
				   _locked, _pmax, _tolerance);
    }
    else if(optimizer == Optimize_FM_MinNetCutMinCut) {
      return FM_MinNetCutMinCut_optimize(_left_index, _left_size,
					 _right_index, _right_size,
					 _locked, _pmax, _tolerance);
    }
    else {
      return 0;
    }
  }

  //
  inline void distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
			 int _left_index, int _left_size, int &_real_left_size,
			 std::deque<int> &_right_stack, std::deque<int> &_right_rank,
			 int _right_index, int _right_size, int &_real_right_size,
			 int _part, std::vector<bool> &_locked,
			 int _current_rank)
  {
    if(distributor == Distribute_Natural_Balanced) {
      NaturalBalanced_distribute(_left_stack, _left_rank,
				 _left_index, _left_size, _real_left_size,
				 _right_stack, _right_rank,
				 _right_index, _right_size, _real_right_size,
				 _part, _locked, _current_rank);
    }
    else if(distributor == Distribute_Natural_Unbalanced) {
      NaturalUnbalanced_distribute(_left_stack, _left_rank,
				   _left_index, _left_size, _real_left_size,
				   _right_stack, _right_rank,
				   _right_index, _right_size, _real_right_size,
				   _part, _locked, _current_rank);
    }
    else if(distributor == Distribute_Randomly) {
      Random_distribute(_left_stack, _left_rank,
			_left_index, _left_size, _real_left_size,
			_right_stack, _right_rank,
			_right_index, _right_size, _real_right_size,
			_part, _locked, _current_rank);
    }
  }


  //The inital distribution
  void NaturalBalanced_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
				  int _left_index, int _left_size, int &_real_left_size,
				  std::deque<int> &_right_stack, std::deque<int> &_right_rank,
				  int _right_index, int _right_size, int &_real_right_size,
				  int _part,
				  std::vector<bool> &_locked,
				  int _current_rank);

  void NaturalUnbalanced_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
				    int _left_index, int _left_size, int &_real_left_size,
				    std::deque<int> &_right_stack, std::deque<int> &_right_rank,
				    int _right_index, int _right_size, int &_real_right_size,
				    int _part,
				    std::vector<bool> &_locked,
				    int _current_rank);

  void Random_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
			 int _left_index, int _left_size, int &_real_left_size,
			 std::deque<int> &_right_stack, std::deque<int> &_right_rank,
			 int _right_index, int _right_size, int &_real_right_size,
			 int _part,
			 std::vector<bool> &_locked,
			 int _current_rank);

  //The different optimzers
  int FM_MinCut_optimize(int _left_index, int _left_size,
			 int _right_index, int _right_size,
			 std::vector<bool> &_locked, int _pmax, int _tolerance);

  int FM_MinNetCut_optimize(int _left_index, int _left_size,
			    int _right_index, int _right_size,
			    std::vector<bool> &_locked, int _pmax, int _tolerance);

  int FM_MinNetCutMinCut_optimize(int _left_index, int _left_size,
				  int _right_index, int _right_size,
				  std::vector<bool> &_locked, int _pmax, int _tolerance);

};

#endif
