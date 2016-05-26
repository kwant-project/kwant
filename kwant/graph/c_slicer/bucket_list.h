// Copyright 2011-2013 Kwant authors.
//
// This file is part of Kwant.  It is subject to the license terms in the file
// LICENSE.rst found in the top-level directory of this distribution and at
// http://kwant-project.org/license.  A list of Kwant authors can be found in
// the file AUTHORS.rst at the top-level directory of this distribution and at
// http://kwant-project.org/authors.

#ifndef BUCKET_LIST_H
#define BUCKET_LIST_H

#include <vector>
#include <list>
#include <utility>
#include <cstdlib>

using std::vector;
using std::list;
using std::pair;
using std::make_pair;

class bucket_list
{
 private:
  vector<list<int> > bucket;
  int max_value;
  int lower_bound, upper_bound;

  vector<list<int>::iterator > &reference_list;
  const vector<int> &index_list;

 public:
  bucket_list(int _lower_bound, int _upper_bound,
              vector<list<int>::iterator > &_reference_list,
              const vector<int> &_index_list) :
    bucket(_upper_bound-_lower_bound+2, list<int>(0)),
    max_value(_lower_bound-1),
    lower_bound(_lower_bound), upper_bound(_upper_bound),
    reference_list(_reference_list), index_list(_index_list)
  {
    //note that the vector bucket also contains an entry
    //for lower_bound-1 !
    //and we fill it with the default variable "-1"
    //so that the past the end bucket is never empty.
    //That makes the algorithms simpler
    bucket[0].push_back(-1);
  }

  inline bool empty() const
  {
    return max_value<lower_bound;
  }

  inline int front() const
  {
    return bucket[max_value - lower_bound + 1].front();
  }

  inline void pop_front()
  {
    if(!empty()) {
      bucket[max_value - lower_bound + 1].pop_front();

      if(bucket[max_value - lower_bound + 1].empty()) {
        while(bucket[--max_value - lower_bound + 1].empty())
          ;
      }
    }
  }

  inline int max_key() const
  {
    return max_value;
  }

  inline void push_back(int _key, int _data)
  {
    reference_list[index_list[_data]]=bucket[_key - lower_bound + 1].
      insert(bucket[_key - lower_bound + 1].end(), _data);

    if(_key > max_value) max_value=_key;
  }

  inline void push_front(int _key, int _data)
  {
    reference_list[index_list[_data]]=bucket[_key - lower_bound + 1].
      insert(bucket[_key - lower_bound + 1].begin(), _data);

    if(_key > max_value) max_value=_key;
  }

  inline void push_randomly(int _key, int _data)
  {
    //do a push_back() or a push_front with equal
    //probability
    if(rand()%2) {
      push_front(_key, _data);
    }
    else {
      push_back(_key, _data);
    }
  }

  inline void rearrange_back(int _old_key, int _new_key, int _data)
  {
    bucket[_old_key - lower_bound +1].
      erase(reference_list[index_list[_data]]);

    reference_list[index_list[_data]]=bucket[_new_key - lower_bound + 1].
      insert(bucket[_new_key - lower_bound + 1].end(), _data);

    if(_new_key > max_value) {
      max_value=_new_key;
    }
    else if(bucket[max_value - lower_bound + 1].empty()) {
      while(bucket[--max_value - lower_bound + 1].empty())
        ;
    }
  }

  inline void rearrange_front(int _old_key, int _new_key, int _data)
  {
    bucket[_old_key - lower_bound +1].
      erase(reference_list[index_list[_data]]);

    reference_list[index_list[_data]]=bucket[_new_key - lower_bound + 1].
      insert(bucket[_new_key - lower_bound + 1].begin(), _data);

    if(_new_key > max_value) {
      max_value=_new_key;
    }
    else if(bucket[max_value - lower_bound + 1].empty()) {
      while(bucket[--max_value - lower_bound + 1].empty())
        ;
    }
  }

  inline void rearrange_randomly(int _old_key, int _new_key, int _data)
  {
    if(rand()%2)
      rearrange_back(_old_key, _new_key, _data);
    else
      rearrange_front(_old_key, _new_key, _data);
  }

  inline void remove(int _key, int _data)
  {
    bucket[_key - lower_bound +1].
      erase(reference_list[index_list[_data]]);

    if(bucket[max_value - lower_bound + 1].empty()) {
      while(bucket[--max_value - lower_bound + 1].empty())
        ;
    }
  }

 private:
  inline list<int>::iterator manual_push_back(int _key, int _data)
  {
    if(_key > max_value) max_value=_key;

    return bucket[_key - lower_bound + 1].
      insert(bucket[_key - lower_bound + 1].end(), _data);
  }

  friend class double_bucket_list;
};

//-------------------

class double_bucket_list
{
 private:
  vector<bucket_list> bucket;

  int max_value;
  int lower_bound, upper_bound;

  vector<list<int>::iterator > &reference_list;
  const vector<int> &index_list;

 public:
  double_bucket_list(int _lower_bound1, int _upper_bound1,
                     int _lower_bound2, int _upper_bound2,
                     vector<list<int>::iterator > &_reference_list,
                     const vector<int> &_index_list) :
    bucket(_upper_bound1-_lower_bound1+2,
           bucket_list(_lower_bound2, _upper_bound2,
           _reference_list, _index_list)),
    max_value(_lower_bound1-1),
    lower_bound(_lower_bound1), upper_bound(_upper_bound1),
    reference_list(_reference_list), index_list(_index_list)
  {
    //note that the vector bucket also contains an entry
    //for lower_bound-1 !
    //and we push a default entry into the past the
    //end bucket of the corresponding bucket_list
    bucket[0].manual_push_back(_lower_bound2, -1);
  }

  inline bool empty()
  {
    return max_value < lower_bound;
  }

  inline int front() const
  {
    return bucket[max_value- lower_bound + 1].front();
  }

  inline void pop_front()
  {
    if(!empty()) {
      bucket[max_value - lower_bound + 1].pop_front();

      if(bucket[max_value - lower_bound + 1].empty()) {
        while(bucket[--max_value - lower_bound + 1].empty())
          ;
      }
    }
  }

  inline pair<int,int> max_key() const
  {
    return make_pair(max_value, bucket[max_value - lower_bound + 1].max_key());
  }

  inline void push_back(int _key1, int _key2, int _data)
  {
    bucket[_key1 - lower_bound + 1].push_back(_key2, _data);

    if(_key1 > max_value) max_value=_key1;
  }

  inline void push_front(int _key1, int _key2, int _data)
  {
    bucket[_key1 - lower_bound + 1].push_front(_key2, _data);

    if(_key1 > max_value) max_value=_key1;
  }

  inline void push_randomly(int _key1, int _key2, int _data)
  {
    if(rand()%2)
      push_back(_key1, _key2, _data);
    else
      push_front(_key1, _key2, _data);
  }

  inline void rearrange_back(int _old_key1, int _old_key2,
                             int _new_key1, int _new_key2, int _data)
  {
    if(_old_key1 == _new_key1) {
      bucket[_old_key1 - lower_bound +1].rearrange_back(_old_key2, _new_key2, _data);
    }
    else {
      bucket[_old_key1 - lower_bound +1].remove(_old_key2, _data);
      bucket[_new_key1 - lower_bound +1].push_back(_new_key2, _data);

      if(_new_key1 > max_value) {
        max_value=_new_key1;
      }
      else if(bucket[max_value - lower_bound + 1].empty()) {
        while(bucket[--max_value - lower_bound + 1].empty())
          ;
      }
    }
  }

  inline void rearrange_front(int _old_key1, int _old_key2,
                              int _new_key1, int _new_key2, int _data)
  {
    if(_old_key1 == _new_key1) {
      bucket[_old_key1 - lower_bound +1].rearrange_front(_old_key2, _new_key2, _data);
    }
    else {
      bucket[_old_key1 - lower_bound +1].remove(_old_key2, _data);
      bucket[_new_key1 - lower_bound +1].push_front(_new_key2, _data);

      if(_new_key1 > max_value) {
        max_value=_new_key1;
      }
      else if(bucket[max_value - lower_bound + 1].empty()) {
        while(bucket[--max_value - lower_bound + 1].empty())
          ;
      }
    }
  }

  inline void rearrange_randomly(int _old_key1, int _old_key2,
                                 int _new_key1, int _new_key2, int _data)
  {
    if(rand()%2)
      rearrange_back(_old_key1, _old_key2, _new_key1, _new_key2, _data);
    else
      rearrange_front(_old_key1, _old_key2, _new_key1, _new_key2, _data);
  }
};

#endif
