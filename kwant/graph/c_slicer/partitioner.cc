#include "partitioner.h"

//----------------------------------------------------
//Functions that do the bisection
//----------------------------------------------------

void Partitioner::bisectFirst(std::vector<int> &_left, std::vector<int> &_right,
			      double _tolerance,
			      int _min_opt_passes,
			      int _max_opt_passes)
{
#if DEBUG_LEVEL>=3
  cout << "bisectFirst" << endl;
#endif

  //First, determine the total number of slices
  int slices=0;

  if(_left.size() && _right.size()) {
    //a starting block is defined on the left and on the right
    std::deque<int> vertex_stack(_left.begin(), _left.end());
    std::deque<int> rank_stack(_left.size(), 1);

    std::vector<bool> locked( parts[0].size(), false);
    std::vector<bool> rightside( parts[0].size(), false);

    std::vector<int>::const_iterator vtx=_right.begin();
    while(vtx!=_right.end()) {
      rightside[*vtx]=true;

      vtx++;
    }

    bool done=false;

    int current_rank=0;

    //mark all the border vertices as visited
    std::deque<int>::const_iterator border_vertex = vertex_stack.begin();
    while(border_vertex != vertex_stack.end()) {
      locked[sliceIndex[*border_vertex]]=true;

      border_vertex++;
    }

    while(vertex_stack.size() && !done ) {
      int vertex = vertex_stack.front();
      int rank   = rank_stack.front();

      vertex_stack.pop_front();
      rank_stack.pop_front();

      //Have we reached a new slice?
      if(rank > current_rank) {
	slices++;
	current_rank=rank;
      }

      //visit the edges
      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {
	//Is the node inside the central slice or already at the right
	//border?
	if(rightside[*edge]) {
	  done=true;
	  break;
	}

	if(!locked[sliceIndex[*edge]]) {
	  vertex_stack.push_back(*edge);
	  rank_stack.push_back(rank+1);

	  locked[sliceIndex[*edge]]=true;
	}

	edge++;
      }
    }

    slices++;

#if DEBUG_LEVEL>=3
    cout << "slices: " << slices << endl;
#endif
  }
  else if(_left.size() || _right.size()) {
    //TODO, not implemented yet
  }
  else {
    int left_border_vertex, right_border_vertex;

    //TODO, not implemented yet
    //slices=findPseudoDiameter(graph, left_border_vertex, right_border_vertex);

#if DEBUG_LEVEL>=3
    cout << "slices: " << slices << endl;
#endif

    _left.push_back(left_border_vertex);
    _right.push_back(right_border_vertex);
  }

#if DEBUG_LEVEL>=3
  cout << "BTD: Total number of slices = " << slices << endl;
#endif

  //Return if there is only one slice
  if(slices<2) return;

  //Find the maximum number of edges (needed or Fiduccia-Mattheyses)
  int max_edges=0;

  for(size_t ivertex=0; ivertex < parts[0].size(); ivertex++) {
    if(graph.getEdges(parts[0][ivertex]).size() > max_edges) {
      max_edges=graph.getEdges(parts[0][ivertex]).size();
    }
  }

  //insert enough space for the parts arrays
  parts.insert(parts.begin()+1, slices-1, std::vector<int>(0));

  //now do the bisection
  bisect(0, slices,
	 _left, _right,
	 max_edges, _tolerance,
	 _min_opt_passes, _max_opt_passes);
}

//-------------

void Partitioner::bisect(int _part, int _part_slices,
			 vector<int> &_left_border,
			 vector<int> &_right_border,
			 int _max_edges, double _tolerance,
			 int _min_opt_passes, int _max_opt_passes)
{
  std::vector<bool> permanently_locked(parts[_part].size(),false);
  std::vector<bool> locked(parts[_part].size(),false);

  //the BFS searches start from the border vertices
  std::deque<int> left_stack(_left_border.begin(), _left_border.end()),
    right_stack(_right_border.begin(), _right_border.end());
  std::deque<int> left_rank(_left_border.size(), 1),
    right_rank(_right_border.size(), 1);

  //The number of slices is already given

  int slices=_part_slices;

#if DEBUG_LEVEL>=3
  cout << "bisect: Total number of slices = " << slices << endl;
#endif

  //Return if there is only one slice
  //(should not happen)
  if(slices<2) return;

  //determine the sizes of the two parts in slices
  int left_slices=slices/2;
  int right_slices=slices-left_slices;

  //calculate the projected slice sizes
  int left_size=left_slices*parts[_part].size()/slices;
  int right_size=parts[_part].size()-left_size;

  int real_left_size=0, real_right_size=0;

  //compute the new indexes
  //according to the index of the first slice in the part
  int left_index=_part;
  int right_index=left_index+left_slices;

#if DEBUG_LEVEL >= 3
  cout << "New indices " << left_index << " " << right_index << endl;
  cout << "New sizes " << left_size << " " << right_size << endl;
#endif

  //Now do a breadth first search from both sides

  //mark all the border vertices as locked and ineligible
  std::deque<int>::const_iterator border_vertex;

  border_vertex=left_stack.begin();
  while(border_vertex != left_stack.end()) {
    locked[sliceIndex[*border_vertex]]=true;
    permanently_locked[sliceIndex[*border_vertex]]=true;

    border_vertex++;
  }

  border_vertex=right_stack.begin();
  while(border_vertex != right_stack.end()) {
    locked[sliceIndex[*border_vertex]]=true;
    permanently_locked[sliceIndex[*border_vertex]]=true;

    border_vertex++;
  }

  //  cout << "Found border" << endl;

  int max_rank=1;

  while(left_stack.size() || right_stack.size()) {

    //first from right
    if(max_rank>right_slices) {
      break;
    }

    while(right_stack.size()) {
      int vertex=right_stack.front();
      int rank=right_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      right_stack.pop_front();
      right_rank.pop_front();

      inSlice[vertex]=right_index;
      real_right_size++;

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !locked[sliceIndex[*edge]]) {

	  right_stack.push_back(*edge);
	  right_rank.push_back(rank+1);

	  locked[sliceIndex[*edge]]=true;
	  if(rank < right_slices) {
	    permanently_locked[sliceIndex[*edge]]=true;
	  }
	}

	edge++;
      }
    }

    //then from left
    if(max_rank>left_slices) {
      break;
    }

    while(left_stack.size()) {
      int vertex=left_stack.front();
      int rank=left_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      left_stack.pop_front();
      left_rank.pop_front();

      inSlice[vertex]=left_index;
      real_left_size++;

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !locked[sliceIndex[*edge]]) {

	  left_stack.push_back(*edge);
	  left_rank.push_back(rank+1);

	  locked[sliceIndex[*edge]]=true;

	  if(rank < left_slices) {
	    permanently_locked[sliceIndex[*edge]]=true;
	  }
	}

	edge++;
      }
    }

    //next slices
    max_rank++;
  }

  distribute(left_stack, left_rank,
	     left_index, left_size, real_left_size,
	     right_stack, right_rank,
	     right_index, right_size, real_right_size,
	     _part, locked, max_rank);

  //Now optimize the structure according to Fiduccia-Mattheyses
  for(int i=0; i<_max_opt_passes; i++) {
    locked=permanently_locked;

    int result=optimize(left_index, left_size, right_index, right_size,
			locked, _max_edges,
			static_cast<int>(_tolerance*parts[left_index].size()));

    if(!result ||
       (i >= _min_opt_passes && result == 1) ) {
      break;
    }
  }

  //create new part arrays
  {
    std::vector<int> left_part, right_part;

    for(size_t ivertex=0; ivertex < parts[left_index].size(); ivertex++) {
      int vertex=parts[left_index][ivertex];

      if(inSlice[vertex]==left_index) {
	left_part.push_back(vertex);
      }
      else {
	right_part.push_back(vertex);
      }
    }

    parts[left_index].swap(left_part);
    parts[right_index].swap(right_part);
  }

  //Now update the sliceIndex
  for(size_t ivertex=0; ivertex < parts[left_index].size(); ivertex++) {
    sliceIndex[parts[left_index][ivertex]]=ivertex;
  }
  for(size_t ivertex=0; ivertex < parts[right_index].size(); ivertex++) {
    sliceIndex[parts[right_index][ivertex]]=ivertex;
  }

  //find the internal borders
  vector<int> internal_left_border, internal_right_border;

  for(size_t ivertex=0; ivertex < parts[left_index].size(); ivertex++) {
    int vertex=parts[left_index][ivertex];

    int *edge=graph.getEdges(vertex).begin();
    while(edge != graph.getEdges(vertex).end()) {

      if(inSlice[*edge]==right_index) {
	internal_left_border.push_back(vertex);

	break;
      }

      edge++;
    }
  }

  for(size_t ivertex=0; ivertex < parts[right_index].size(); ivertex++) {
    int vertex=parts[right_index][ivertex];

    int *edge=graph.getEdges(vertex).begin();
    while(edge != graph.getEdges(vertex).end()) {

      if(inSlice[*edge]==left_index) {
	internal_right_border.push_back(vertex);

	break;
      }

      edge++;
    }
  }


  /*  //debug
      ostringstream convert;
      convert << "part" << counter++ << ".eps";
      write2DToEPS(convert.str().c_str());

      cout << "Test " <<counter << " " << parts[left_index].size() << " " << parts[right_index].size() << endl;

  */
  //Recursively refine the bisection
  if(left_slices>1) {
    bisect(left_index, left_slices,
	   _left_border, internal_left_border,
	   _max_edges, _tolerance,
	   _min_opt_passes, _max_opt_passes);
  }
  if(right_slices>1) {
    bisect(right_index, right_slices,
	   internal_right_border, _right_border,
	   _max_edges, _tolerance,
	   _min_opt_passes, _max_opt_passes);
  }
}

//-------------------------------------------------------
//The initial distributions
//-------------------------------------------------------

void Partitioner::
NaturalBalanced_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
			   int _left_index, int _left_size, int &_real_left_size,
			   std::deque<int> &_right_stack, std::deque<int> &_right_rank,
			   int _right_index, int _right_size, int &_real_right_size,
			   int _part, std::vector<bool> &_locked,
			   int _current_rank)
{
  int max_rank=_current_rank;

  while(_left_stack.size() || _right_stack.size()) {

    //first from right
    while(_right_stack.size()) {
      int vertex=_right_stack.front();
      int rank=_right_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _right_stack.pop_front();
      _right_rank.pop_front();

      //Check if we have already exceeded the size of this
      //part of the bisection
      if(_real_right_size >= _right_size) {
	inSlice[vertex]=_left_index;
	_real_left_size++;
      }
      else {
	inSlice[vertex]=_right_index;
	_real_right_size++;
      }

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _right_stack.push_back(*edge);
	  _right_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;
	}

	edge++;
      }
    }

    //then from left
    while(_left_stack.size()) {
      int vertex=_left_stack.front();
      int rank=_left_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _left_stack.pop_front();
      _left_rank.pop_front();

      //Check if we have already exceeded the size of this
      //part of the bisection
      if(_real_left_size >= _left_size) {
	inSlice[vertex]=_right_index;
	_real_right_size++;
      }
      else {
	inSlice[vertex]=_left_index;
	_real_left_size++;
      }

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _left_stack.push_back(*edge);
	  _left_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;

	}

	edge++;
      }
    }

    //next slices
    max_rank++;
  }

}

void Partitioner::
NaturalUnbalanced_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
			     int _left_index, int _left_size, int &_real_left_size,
			     std::deque<int> &_right_stack, std::deque<int> &_right_rank,
			     int _right_index, int _right_size, int &_real_right_size,
			     int _part, std::vector<bool> &_locked,
			     int _current_rank)
{
  int max_rank=_current_rank;

  while(_left_stack.size() || _right_stack.size()) {

    //first from right
    while(_right_stack.size()) {
      int vertex=_right_stack.front();
      int rank=_right_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _right_stack.pop_front();
      _right_rank.pop_front();

      inSlice[vertex]=_right_index;
      _real_right_size++;

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _right_stack.push_back(*edge);
	  _right_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;
	}

	edge++;
      }
    }

    //then from left
    while(_left_stack.size()) {
      int vertex=_left_stack.front();
      int rank=_left_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _left_stack.pop_front();
      _left_rank.pop_front();

      inSlice[vertex]=_left_index;
      _real_left_size++;

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _left_stack.push_back(*edge);
	  _left_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;

	}

	edge++;
      }
    }

    //next slices
    max_rank++;
  }

}

void Partitioner::
Random_distribute(std::deque<int> &_left_stack, std::deque<int> &_left_rank,
		  int _left_index, int _left_size, int &_real_left_size,
		  std::deque<int> &_right_stack, std::deque<int> &_right_rank,
		  int _right_index, int _right_size, int &_real_right_size,
		  int _part, std::vector<bool> &_locked,
		  int _current_rank)
{
  int max_rank=_current_rank;

  while(_left_stack.size() || _right_stack.size()) {

    //first from right
    while(_right_stack.size()) {
      int vertex=_right_stack.front();
      int rank=_right_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _right_stack.pop_front();
      _right_rank.pop_front();

      if(rand()%2) {
	inSlice[vertex]=_right_index;
	_real_right_size++;
      }
      else {
	inSlice[vertex]=_left_index;
	_real_left_size++;
      }

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _right_stack.push_back(*edge);
	  _right_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;
	}

	edge++;
      }
    }

    //then from left
    while(_left_stack.size()) {
      int vertex=_left_stack.front();
      int rank=_left_rank.front();

      //stop if we have exceeded the desired rank
      if(rank>max_rank) {
	break;
      }

      _left_stack.pop_front();
      _left_rank.pop_front();

      if(rand()%2) {
	inSlice[vertex]=_left_index;
	_real_left_size++;
      }
      else {
	inSlice[vertex]=_right_index;
	_real_right_size++;
      }

      int *edge=graph.getEdges(vertex).begin();
      while(edge != graph.getEdges(vertex).end()) {

	if(inSlice[*edge]==_part &&
	   !_locked[sliceIndex[*edge]]) {

	  _left_stack.push_back(*edge);
	  _left_rank.push_back(rank+1);

	  _locked[sliceIndex[*edge]]=true;

	}

	edge++;
      }
    }

    //next slices
    max_rank++;
  }
}

//-------------------------------------------------------
//Optimizers
//-------------------------------------------------------

int Partitioner::
FM_MinCut_optimize(int _left_index, int _left_size,
		   int _right_index, int _right_size,
		   std::vector<bool> &_locked, int _pmax, int _tolerance)
{
  std::vector<int> gain(_left_size+_right_size, 0);

  std::vector<list<int>::iterator> vertex_list(_left_size+_right_size);
  std::vector<bucket_list> bucket(2, bucket_list(-_pmax, _pmax,
						 vertex_list, sliceIndex));

  std::vector<int> change_buffer;
  int size[2];
  int part_index[2];

  part_index[0]=_left_index;
  part_index[1]=_right_index;

  //relative sizes and tolerances

  size[0] = -_left_size;
  size[1] = -_right_size;

  int tolerance=std::max(1, _tolerance);
  //In order for Fiduccia-Mattheyses to work,
  //we may at least allow an imbalance of 1 vertex

#if DEBUG_LEVEL>=3
  cout << _left_size << " " << _right_size << " " << parts[_left_index].size() << endl;
#endif

  //Initialize gain values

  //first left part
  for(size_t ivertex=0; ivertex < parts[_left_index].size(); ivertex++) {
    int index, part_index;
    int vertex=parts[_left_index][ivertex];

    if(inSlice[vertex] == _left_index) {
      part_index=_left_index;
      index=0;
    }
    else {
      part_index=_right_index;
      index=1;
    }

    //Update the partition size
    size[index]++;

    //For all the free vertices we go through

    //all the edges to compute the gain
    if(!_locked[sliceIndex[vertex]]) {
      int *edge=graph.getEdges(vertex).begin();

      while(edge != graph.getEdges(vertex).end()) {
	if(inSlice[*edge]!=part_index) {
	  gain[sliceIndex[vertex]]++;
	}
	else {
	  gain[sliceIndex[vertex]]--;
	}

	edge++;
      }

      //Finally add the vertex to the bucket list
      bucket[index].push_back(gain[sliceIndex[vertex]], vertex);
    }
  }

#if DEBUG_LEVEL>=3
  cout << size[0] << " " << size[1] << endl;
#endif

  //characteristics of the best partition
  int relative_gain=0;

  int old_size=abs(size[0]);
  //int old_gain not needed, -> relative gain!

  int best_gain=0;
  int best_size=abs(size[0]);

  int best_move=0;

  while(true) {
    //Choose the cell that should be moved
    int from, to;

    int cur_gain[2];

    cur_gain[0]=bucket[0].max_key();
    cur_gain[1]=bucket[1].max_key();

    //Only allow moves that improve or maintain balance:
    //check if partition is imbalanced or could be imbalanced
    //at the next move
    //(Note: it doesn't matter whether we compare size[0] or
    // size[1] with tolerance, since size[0]+size[1]=const
    if(abs(size[0])>=tolerance) {
      if(size[0] > size[1]) {
	cur_gain[1]=-_pmax-1;
      }
      else {
	cur_gain[0]=-_pmax-1;
      }
    }

    //Choose the cell with the largest gain
    //(Note: moves that could cause imbalance
    // are prevented by the previous checks)
    if(cur_gain[0] != cur_gain[1]) {
      from=(cur_gain[0] > cur_gain[1]) ? 0 : 1;
      to=(cur_gain[0] > cur_gain[1]) ? 1 : 0;
    }
    //if the gains are equal, check if no further
    //moves are possible, otherwise choose
    //the move that improves balance, if it doesn't matter
    //take the left slice
    else {
      if(cur_gain[0] > -_pmax-1) {
	from=(size[0] >= size[1]) ? 0 : 1;
	to=(size[0] >= size[1]) ? 1 : 0;
      }
      else {
	//no further moves are possible

	break;
      }
    }

    //Remove the vertex and adjust partition size
    int vertex=bucket[from].front();
    bucket[from].pop_front();

    _locked[sliceIndex[vertex]]=true;
    inSlice[vertex]=part_index[to];

    size[from]--;
    size[to]++;

    relative_gain+=cur_gain[from];
    //    relative_gain+=gain[sliceIndex[vertex]];

    change_buffer.push_back(vertex);

    //cout << "Moving vertex " << vertex
    //    << " with gain " << cur_gain[from] << endl;

    //update gains and adjust bucket structure
    int *edge=graph.getEdges(vertex).begin();
    while(edge != graph.getEdges(vertex).end()) {

      if(inSlice[*edge]==part_index[from]) {
	if(!_locked[sliceIndex[*edge]]) {
	  int old_gain=gain[sliceIndex[*edge]];

	  gain[sliceIndex[*edge]]+=2;

	  bucket[from].rearrange_back(old_gain, old_gain+2, *edge);
	}
      }
      else if(inSlice[*edge]==part_index[to]) {
	if(!_locked[sliceIndex[*edge]]) {
	  int old_gain=gain[sliceIndex[*edge]];

	  gain[sliceIndex[*edge]]-=2;

	  bucket[to].rearrange_back(old_gain, old_gain-2, *edge);
	}
      }
      edge++;
    }


    //Have we found a better partition
    if(relative_gain > best_gain ||
       (relative_gain == best_gain && best_size > abs(size[0]))) {
      best_gain=relative_gain;
      best_size=abs(size[0]);

      best_move=change_buffer.size();
    }

  }

  //Undo all the changes that did not improve the
  //bisection any more:
  for(size_t ivertex=best_move; ivertex < change_buffer.size(); ivertex++) {
    int vertex=change_buffer[ivertex];

    if(inSlice[vertex] == _left_index) {
      inSlice[vertex]=_right_index;
    }
    else {
      inSlice[vertex]=_left_index;
    }
  }

  if(best_move == 0) {
    return 0; //no improvements possible
  }
  else {
    if(best_gain > 0 ||
       (best_gain == 0 && best_size < old_size) ){
      return 2; //definite improvement
    }
    else {
      return 1; //found a partition with the same properties as before
                //(needed to get out of local minima)
    }
  }
}


//----------------

int Partitioner::
FM_MinNetCut_optimize(int _left_index, int _left_size,
		      int _right_index, int _right_size,
		      std::vector<bool> &_locked, int _pmax, int _tolerance)
{
#if DEBUG_LEVEL>=3
  cout << "In FM MinNetCut" << endl;
#endif

  std::vector<int> net_gain(_left_size+_right_size, 0);
  std::vector<int> edge_gain(_left_size+_right_size, 0);

  std::vector<int> net_distribution[2];

  std::vector<list<int>::iterator> vertex_list(_left_size+_right_size);
  std::vector<bucket_list> bucket(2, bucket_list(-_pmax, _pmax,
						 vertex_list, sliceIndex));

  std::vector<int> change_buffer;
  int size[2];
  int part_index[2];

  net_distribution[0].resize(_left_size+_right_size);
  net_distribution[1].resize(_left_size+_right_size);

  part_index[0]=_left_index;
  part_index[1]=_right_index;

  //relative sizes and tolerances

  size[0] = -_left_size;
  size[1] = -_right_size;

  int tolerance=std::max(1, _tolerance);
  //In order for Fiduccia-Mattheyses to work,
  //we may at least allow an imbalance of 1 vertex

#if DEBUG_LEVEL>=3
  cout << _left_size << " " << _right_size << " " << parts[_left_index].size() << endl;
#endif

  //Initialize the distribution of the nets
  for(size_t ivertex=0; ivertex < parts[_left_index].size(); ivertex++) {
    int vertex=parts[_left_index][ivertex];

    net_distribution[0][sliceIndex[vertex]]=0;
    net_distribution[1][sliceIndex[vertex]]=0;

    //the net includes the node itself ...
    if(inSlice[vertex] == _left_index) {
      net_distribution[0][sliceIndex[vertex]]++;
    }
    else {
      net_distribution[1][sliceIndex[vertex]]++;
    }

    //and it's nearest neighbours
    int *edge=graph.getEdges(vertex).begin();

    while(edge != graph.getEdges(vertex).end()) {

      if(inSlice[*edge] <= _left_index) {
	net_distribution[0][sliceIndex[vertex]]++;
      }
      else {
	net_distribution[1][sliceIndex[vertex]]++;
      }

      edge++;
    }
  }

  //Initialize gain values
  for(size_t ivertex=0; ivertex < parts[_left_index].size(); ivertex++) {
    int index, part_index;
    int vertex=parts[_left_index][ivertex];
    int from,to;

    if(inSlice[vertex] == _left_index) {
      part_index=_left_index;
      index=0;
      from=0;to=1;
    }
    else {
      part_index=_right_index;
      index=1;
      from=1;to=0;
    }

    //Update the partition size
    size[index]++;

    //For all the free vertices we go through

    //all the edges to compute the gain
    if(!_locked[sliceIndex[vertex]]) {
      int *edge=graph.getEdges(vertex).begin();

      while(edge != graph.getEdges(vertex).end()) {

	//Update the gain with regard to cut edges
	if(inSlice[*edge]!=part_index) {
	  edge_gain[sliceIndex[vertex]]++;
	}
	else {
	  edge_gain[sliceIndex[vertex]]--;	}

	//and with regard to cut nets
	if(net_distribution[from][sliceIndex[*edge]] == 1) {
	  net_gain[sliceIndex[vertex]]++;
	}
	else if(net_distribution[to][sliceIndex[*edge]] == 0) {
	  net_gain[sliceIndex[vertex]]--;
	}

	edge++;
      }

      //Finally add the vertex to the bucket list
      bucket[index].push_randomly(net_gain[sliceIndex[vertex]], vertex);

    }
  }

#if DEBUG_LEVEL>=3
  cout << size[0] << " " << size[1] << endl;
#endif

  //characteristics of the best partition
  int relative_gain=0;
  int relative_edge_gain=0;

  int old_size=abs(size[0]);
  //int old_gains not needed -> relative gains!

  int best_gain=0;
  int best_edge_gain=0;
  int best_size=abs(size[0]);

  int best_move=0;

  while(true) {
    //Choose the cell that should be moved
    int from, to;

    int cur_gain[2];

    cur_gain[0]=bucket[0].max_key();
    cur_gain[1]=bucket[1].max_key();

    //Only allow moves that improve or maintain balance:
    //check if partition is imbalanced or could be imbalanced
    //at the next move
    //(Note: it doesn't matter whether we compare size[0] or
    // size[1] with tolerance, since size[0]+size[1]=const
    if(abs(size[0])>=tolerance) {
      if(size[0] > size[1]) {
	cur_gain[1]=-_pmax-1;
      }
      else {
	cur_gain[0]=-_pmax-1;
      }
    }

    //Choose the cell with the largest gain
    //(Note: moves that could cause imbalance
    // are prevented by the previous checks)
    if(cur_gain[0] != cur_gain[1]) {
      from=(cur_gain[0] > cur_gain[1]) ? 0 : 1;
      to=(cur_gain[0] > cur_gain[1]) ? 1 : 0;
    }
    //if the gains are equal, check if no further
    //moves are possible, otherwise choose
    //the move that improves balance, if it doesn't matter
    //take the left slice
    else {
      if(cur_gain[0] > -_pmax-1) {
	from=(size[0] >= size[1]) ? 0 : 1;
	to=(size[0] >= size[1]) ? 1 : 0;
      }
      else {
	//no further moves are possible

	break;
      }
    }

    //Remove the vertex and adjust partition size
    int vertex=bucket[from].front();
    bucket[from].pop_front();

    _locked[sliceIndex[vertex]]=true;
    inSlice[vertex]=part_index[to];

    size[from]--;
    size[to]++;

    relative_gain+=cur_gain[from];
    relative_edge_gain+=edge_gain[sliceIndex[vertex]];
    change_buffer.push_back(vertex);

    //update gains and adjust bucket structure
    int *edge=graph.getEdges(vertex).begin();
    while(edge != graph.getEdges(vertex).end()) {

      //----------------
      //update net gains
      //----------------

      if(net_distribution[to][sliceIndex[*edge]] == 0) {
	if(!_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]++;

	  //Note: all the vertices are in the to part
	  bucket[from].rearrange_randomly(old_gain, old_gain+1, *edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(!_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]++;

	    //Note: all the vertices are in the from part
	    bucket[from].rearrange_randomly(old_gain, old_gain+1, *edgedge);
	  }
	  edgedge++;
	}
      }
      else if(net_distribution[to][sliceIndex[*edge]] == 1) {
	if(inSlice[*edge]==part_index[to] && !_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]--;

	  //Note: all the vertices are in the to part
	  bucket[to].rearrange_randomly(old_gain, old_gain-1, *edge);

	}
	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(inSlice[*edgedge]==part_index[to] && !_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]--;

	    bucket[to].rearrange_randomly(old_gain, old_gain-1, *edgedge);

	    break; //there is only one vertex in the to part
	    //(well, it's two after the move, nut the moved vertex is locked)
	  }

	  edgedge++;
	}
      }

      net_distribution[from][sliceIndex[*edge]]--;
      net_distribution[to][sliceIndex[*edge]]++;

      if(net_distribution[from][sliceIndex[*edge]] == 0) {
	if(!_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]--;

	  //Note: all the vertices are in the to part
	  bucket[to].rearrange_randomly(old_gain, old_gain-1, *edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(!_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]--;

	    bucket[to].rearrange_randomly(old_gain, old_gain-1, *edgedge);
	  }
	  edgedge++;
	}

      }
      else if(net_distribution[from][sliceIndex[*edge]] == 1) {
	if(inSlice[*edge]==part_index[from] && !_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]++;

	  bucket[from].rearrange_randomly(old_gain, old_gain+1, *edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(inSlice[*edgedge]==part_index[from] && !_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]++;

	    bucket[from].rearrange_randomly(old_gain, old_gain+1, *edgedge);
	    break; //there is only one vertex in the from part
	    //(the other one has been moved)
	  }

	  edgedge++;
	}
      }

      //-----------------
      //update edge gains
      //-----------------

      if(inSlice[*edge]==part_index[from]) {
	if(!_locked[sliceIndex[*edge]]) {
	  edge_gain[sliceIndex[*edge]]+=2;
	}
      }
      else if(inSlice[*edge]==part_index[to]) {
	if(!_locked[sliceIndex[*edge]]) {
	  edge_gain[sliceIndex[*edge]]-=2;
	}
      }

      edge++;
    }

    //Have we found a better partition
    if(relative_gain > best_gain ||
       (relative_gain == best_gain && best_size >= abs(size[0]) ) ||
       (relative_gain == best_gain && best_size == abs(size[0]) && relative_edge_gain>=best_edge_gain) ) {
      best_gain=relative_gain;
      best_edge_gain=relative_edge_gain;
      best_size=abs(size[0]);

      best_move=change_buffer.size();
    }

  }

  //Undo all the changes that did not improve the
  //bisection any more:
  for(size_t ivertex=best_move; ivertex < change_buffer.size(); ivertex++) {
    int vertex=change_buffer[ivertex];

    if(inSlice[vertex] == _left_index) {
      inSlice[vertex]=_right_index;
    }
    else {
      inSlice[vertex]=_left_index;
    }
  }

  if(best_move == 0) {
    return 0; //no improvements possible
  }
  else {
    if(best_gain > 0 ||
       (best_gain == 0 && best_size < old_size) ||
       (best_gain == 0 && best_size == old_size && best_edge_gain > 0) ){
      return 2; //definite improvement
    }
    else {
      return 1; //found a partition with the same properties as before
                //(needed to get out of local minima)
    }
  }
}

//----------------

int Partitioner::
FM_MinNetCutMinCut_optimize(int _left_index, int _left_size,
			    int _right_index, int _right_size,
			    std::vector<bool> &_locked, int _pmax, int _tolerance)
{
#if DEBUG_LEVEL>=3
  cout << "In FM MinNetCut_MinCut" << endl;
#endif

  std::vector<int> net_gain(_left_size+_right_size, 0);
  std::vector<int> edge_gain(_left_size+_right_size, 0);

  std::vector<int> net_distribution[2];

  std::vector<list<int>::iterator> vertex_list(_left_size+_right_size);
  std::vector<double_bucket_list> bucket(2, double_bucket_list(-_pmax, _pmax,
							       -_pmax, _pmax,
							       vertex_list, sliceIndex));

  std::vector<int> change_buffer;
  int size[2];
  int part_index[2];

  net_distribution[0].resize(_left_size+_right_size);
  net_distribution[1].resize(_left_size+_right_size);

  part_index[0]=_left_index;
  part_index[1]=_right_index;

  //relative sizes and tolerances

  size[0] = -_left_size;
  size[1] = -_right_size;

  int tolerance=std::max(1, _tolerance);
  //In order for Fiduccia-Mattheyses to work,
  //we may at least allow an imbalance of 1 vertex

#if DEBUG_LEVEL>=3
  cout << _left_size << " " << _right_size << " " << parts[_left_index].size() << endl;
#endif

  //Initialize the distribution of the nets
  for(size_t ivertex=0; ivertex < parts[_left_index].size(); ivertex++) {
    int vertex=parts[_left_index][ivertex];

    net_distribution[0][sliceIndex[vertex]]=0;
    net_distribution[1][sliceIndex[vertex]]=0;

    //the net includes the node itself ...
    if(inSlice[vertex] == _left_index) {
      net_distribution[0][sliceIndex[vertex]]++;
    }
    else {
      net_distribution[1][sliceIndex[vertex]]++;
    }

    //and it's nearest neighbours
    int *edge=graph.getEdges(vertex).begin();

    while(edge != graph.getEdges(vertex).end()) {

      if(inSlice[*edge] <= _left_index) {
	net_distribution[0][sliceIndex[vertex]]++;
      }
      else {
	net_distribution[1][sliceIndex[vertex]]++;
      }

      edge++;
    }
  }

  //Initialize gain values
  for(size_t ivertex=0; ivertex < parts[_left_index].size(); ivertex++) {
    int index, part_index;
    int vertex=parts[_left_index][ivertex];
    int from,to;

    if(inSlice[vertex] == _left_index) {
      part_index=_left_index;
      index=0;
      from=0;to=1;
    }
    else {
      part_index=_right_index;
      index=1;
      from=1;to=0;
    }

    //Update the partition size
    size[index]++;

    //For all the free vertices we go through

    //all the edges to compute the gain
    if(!_locked[sliceIndex[vertex]]) {
      int *edge=graph.getEdges(vertex).begin();

      while(edge != graph.getEdges(vertex).end()) {

	//Update the gain with regard to cut edges
	if(inSlice[*edge]!=part_index) {
	  edge_gain[sliceIndex[vertex]]++;
	}
	else {
	  edge_gain[sliceIndex[vertex]]--;
	}

	//and with regard to cut nets
	if(net_distribution[from][sliceIndex[*edge]] == 1) {
	  net_gain[sliceIndex[vertex]]++;
	}
	else if(net_distribution[to][sliceIndex[*edge]] == 0) {
	  net_gain[sliceIndex[vertex]]--;
	}

	edge++;
      }

      //Finally add the vertex to the bucket list
      bucket[index].push_randomly(net_gain[sliceIndex[vertex]],
				  edge_gain[sliceIndex[vertex]], vertex);
    }
  }

#if DEBUG_LEVEL>=3
  cout << "deviation: " << size[0] << " " << size[1] << endl;
#endif

  //characteristics of the best partition
  int relative_gain=0;
  int relative_edge_gain=0;

  int old_size=abs(size[0]);
  //int old_gains not needed -> relative gains!

  int best_gain=0;
  int best_edge_gain=0;
  int best_size=abs(size[0]);

  int best_move=0;

  while(true) {
    //Choose the cell that should be moved
    int from, to;

    int cur_net_gain[2];

    cur_net_gain[0]=bucket[0].max_key().first;
    cur_net_gain[1]=bucket[1].max_key().first;

    //Only allow moves that improve or maintain balance:
    //check if partition is imbalanced or could be imbalanced
    //at the next move
    //(Note: it doesn't matter whether we compare size[0] or
    // size[1] with tolerance, since size[0]+size[1]=const
    if(abs(size[0])>=tolerance) {
      if(size[0] > size[1]) {
	cur_net_gain[1]=-_pmax-1;
      }
      else {
	cur_net_gain[0]=-_pmax-1;
      }
    }

    //Choose the cell with the largest gain
    //(Note: moves that could cause imbalance
    // are prevented by the previous checks)
    if(cur_net_gain[0] != cur_net_gain[1]) {
      from=(cur_net_gain[0] > cur_net_gain[1]) ? 0 : 1;
      to=(cur_net_gain[0] > cur_net_gain[1]) ? 1 : 0;
    }
    //if the gains are equal, check if no further
    //moves are possible, otherwise choose
    //the move that improves balance, if it doesn't matter
    //take the left slice
    else {
      if(cur_net_gain[0] > -_pmax-1) {
	from=(size[0] >= size[1]) ? 0 : 1;
	to=(size[0] >= size[1]) ? 1 : 0;
      }
      else {
	//no further moves are possible

	break;
      }
    }

    //Remove the vertex and adjust partition size
    int vertex=bucket[from].front();
    bucket[from].pop_front();

    _locked[sliceIndex[vertex]]=true;
    inSlice[vertex]=part_index[to];

    size[from]--;
    size[to]++;

    relative_gain+=cur_net_gain[from];
    relative_edge_gain+=bucket[from].max_key().second;

    change_buffer.push_back(vertex);

    //update gains and adjust bucket structure
    int *edge=graph.getEdges(vertex).begin();
    while(edge != graph.getEdges(vertex).end()) {

      //-----------------
      //update edge gains
      //-----------------
      int old_edge_gain=edge_gain[sliceIndex[*edge]];

      if(inSlice[*edge]==part_index[from]) {
	if(!_locked[sliceIndex[*edge]]) {
	  edge_gain[sliceIndex[*edge]]+=2;

	  bucket[from].rearrange_randomly(net_gain[sliceIndex[*edge]], old_edge_gain,
					  net_gain[sliceIndex[*edge]], old_edge_gain+2,
					  *edge);
	}
      }
      else if(inSlice[*edge]==part_index[to]) {
	if(!_locked[sliceIndex[*edge]]) {
	  edge_gain[sliceIndex[*edge]]-=2;

	  bucket[to].rearrange_randomly(net_gain[sliceIndex[*edge]], old_edge_gain,
					net_gain[sliceIndex[*edge]], old_edge_gain-2,
					*edge);
	}
      }

      //----------------
      //update net gains
      //----------------

      if(net_distribution[to][sliceIndex[*edge]] == 0) {
	if(!_locked[sliceIndex[*edge]]) {

	  int old_gain=net_gain[sliceIndex[*edge]]++;

	  //Note: all the vertices are in the from part
	  bucket[from].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edge]],
					  old_gain+1, edge_gain[sliceIndex[*edge]],
					  *edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(!_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]++;

	    //Note: all the vertices are in the from part
	    bucket[from].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edgedge]],
					    old_gain+1, edge_gain[sliceIndex[*edgedge]],
					    *edgedge);
	  }
	  edgedge++;
	}
      }
      else if(net_distribution[to][sliceIndex[*edge]] == 1) {
	if(inSlice[*edge]==part_index[to] && !_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]--;

	  //Note: all the vertices are in the to part
	  bucket[to].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edge]],
					old_gain-1, edge_gain[sliceIndex[*edge]],
					*edge);

	}
	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(inSlice[*edgedge]==part_index[to] && !_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]--;

	    bucket[to].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edgedge]],
					  old_gain-1, edge_gain[sliceIndex[*edgedge]],
					  *edgedge);

	    break; //there is only one vertex in the to part
	    //(well, it's two after the move, nut the moved vertex is locked)
	  }

	  edgedge++;
	}
      }

      net_distribution[from][sliceIndex[*edge]]--;
      net_distribution[to][sliceIndex[*edge]]++;

      if(net_distribution[from][sliceIndex[*edge]] == 0) {
	if(!_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]--;

	  //Note: all the vertices are in the to part
	  bucket[to].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edge]],
					old_gain-1, edge_gain[sliceIndex[*edge]],
					*edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }
	  if(!_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]--;

	    bucket[to].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edgedge]],
					  old_gain-1, edge_gain[sliceIndex[*edgedge]],
					  *edgedge);
	  }
	  edgedge++;
	}

      }
      else if(net_distribution[from][sliceIndex[*edge]] == 1) {
	if(inSlice[*edge]==part_index[from] && !_locked[sliceIndex[*edge]]) {
	  int old_gain=net_gain[sliceIndex[*edge]]++;

	  bucket[from].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edge]],
					  old_gain+1, edge_gain[sliceIndex[*edge]],
					  *edge);
	}

	int *edgedge=graph.getEdges(*edge).begin();
	while(edgedge != graph.getEdges(*edge).end()) {
	  if(inSlice[*edgedge]!=_left_index &&
	     inSlice[*edgedge]!=_right_index) {
	    edgedge++;
	    continue;
	  }

	  if(inSlice[*edgedge]==part_index[from] && !_locked[sliceIndex[*edgedge]]) {
	    int old_gain=net_gain[sliceIndex[*edgedge]]++;

	    bucket[from].rearrange_randomly(old_gain, edge_gain[sliceIndex[*edgedge]],
					    old_gain+1, edge_gain[sliceIndex[*edgedge]],
					    *edgedge);
	    break; //there is only one vertex in the from part
	    //(the other one has been moved)
	  }

	  edgedge++;
	}
      }

      edge++;
    }

    //Have we found a better partition
    if(relative_gain > best_gain ||
       (relative_gain == best_gain && relative_edge_gain >= best_edge_gain) ||
       (relative_gain == best_gain && relative_edge_gain==best_edge_gain && best_size >= abs(size[0]))) {
      best_gain=relative_gain;
      best_edge_gain=relative_edge_gain;
      best_size=abs(size[0]);

      best_move=change_buffer.size();
    }

  }

#if DEBUG_LEVEL>2
  cout << "best_move: " << best_move << endl;
  cout << "best gain: " << best_gain << endl;
  cout << "best size: " << best_size << endl;
#endif

  //Undo all the changes that did not improve the
  //bisection any more:
  for(size_t ivertex=best_move; ivertex < change_buffer.size(); ivertex++) {
    int vertex=change_buffer[ivertex];

    if(inSlice[vertex] == _left_index) {
      inSlice[vertex]=_right_index;
    }
    else {
      inSlice[vertex]=_left_index;
    }
  }

  if(best_move == 0) {
    return 0; //no improvements possible
  }
  else {
    if(best_gain > 0 ||
       (best_gain == 0 && best_size < old_size) ||
       (best_gain == 0 && best_size == old_size && best_edge_gain > 0) ) {
      return 2; //definite improvement
    }
    else {
      return 1; //found a partition with the same properties as before
                //(needed to get out of local minima)
    }
  }
}
