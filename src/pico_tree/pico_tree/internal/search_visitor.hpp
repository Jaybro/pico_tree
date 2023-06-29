#pragma once

#include <algorithm>
#include <iterator>

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

//! \brief Inserts \p item in O(n) time at the index for which \p comp first
//! holds true. The sequence must be sorted and remains sorted after insertion.
//! The last item in the sequence is overwritten / "pushed out".
//! \details The contents of the indices at which \p comp holds true are moved
//! to the next index. Thus, starting from the end of the sequence, each item[i]
//! gets replaced by item[i - 1] until \p comp results in false. The worst case
//! has n comparisons and n copies, traversing the entire sequence.
//! <p/>
//! This algorithm is used as the inner loop of insertion sort:
//! * https://en.wikipedia.org/wiki/Insertion_sort
template <
    typename RandomAccessIterator_,
    typename Compare_ = std::less<
        typename std::iterator_traits<RandomAccessIterator_>::value_type>>
inline void InsertSorted(
    RandomAccessIterator_ begin,
    RandomAccessIterator_ end,
    typename std::iterator_traits<RandomAccessIterator_>::value_type item,
    Compare_ comp = Compare_()) {
  std::advance(end, -1);
  for (; end > begin && comp(item, *std::prev(end)); --end) {
    *end = std::move(*std::prev(end));
  }
  // We update the inserted element outside of the loop. This is done for the
  // case where we didn't break, simply reaching the end of the loop. This
  // happens when we need to replace the first element in the sequence (the last
  // item encountered).
  *end = std::move(item);
}

//! \brief KdTree search visitor for finding a single nearest neighbor.
template <typename Neighbor_>
class SearchNn {
 public:
  using NeighborType = Neighbor_;
  using IndexType = typename Neighbor_::IndexType;
  using ScalarType = typename Neighbor_::ScalarType;

  //! \private
  inline SearchNn(NeighborType& nn) : nn_{nn} {
    nn_.distance = std::numeric_limits<ScalarType>::max();
  }

  //! \brief Visit current point.
  inline void operator()(IndexType const idx, ScalarType const dst) const {
    nn_ = {idx, dst};
  }

  //! \brief Maximum search distance with respect to the query point.
  inline ScalarType max() const { return nn_.distance; }

 private:
  NeighborType& nn_;
};

//! \brief KdTree search visitor for finding k nearest neighbors using an
//! insertion sort.
//! \details Even though insertion sort is a rather brute-force method for
//! maintaining a sorted sequence, the k nearest neighbors, it performs fast in
//! practice. This is likely due to points being reasonably ordered by the
//! KdTree. The following strategies have been attempted:
//!  * std::vector::insert(std::lower_bound) - the predecessor of the current
//!  version.
//!  * std::push_heap(std::vector) and std::pop_heap(std::vector).
//!  * std::push_heap(std::vector) followed by a custom ReplaceFrontHeap once
//!  the heap reached size k. This is the fastest "priority queue" version so
//!  far. Even without sorting the heap it is still slower than maintaining a
//!  sorted sequence. Unsorted it does come close to the insertion sort.
//!  * Binary heap plus a heap sort seemed a lot faster than the Leonardo heap
//!  with smooth sort.
template <typename RandomAccessIterator_>
class SearchKnn {
 public:
  static_assert(
      std::is_base_of_v<
          std::random_access_iterator_tag,
          typename std::iterator_traits<
              RandomAccessIterator_>::iterator_category>,
      "SEARCH_KNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using NeighborType =
      typename std::iterator_traits<RandomAccessIterator_>::value_type;
  using IndexType = typename NeighborType::IndexType;
  using ScalarType = typename NeighborType::ScalarType;

  //! \private
  inline SearchKnn(RandomAccessIterator_ begin, RandomAccessIterator_ end)
      : begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<ScalarType>::max();
  }

  //! \brief Visit current point.
  inline void operator()(IndexType const idx, ScalarType const dst) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    InsertSorted(begin_, active_end_, NeighborType{idx, dst});
  }

  //! \brief Maximum search distance with respect to the query point.
  inline ScalarType max() const { return std::prev(end_)->distance; }

 private:
  RandomAccessIterator_ begin_;
  RandomAccessIterator_ end_;
  RandomAccessIterator_ active_end_;
};

//! \brief KdTree search visitor for finding all neighbors within a radius.
template <typename Neighbor_>
class SearchRadius {
 public:
  using NeighborType = Neighbor_;
  using IndexType = typename Neighbor_::IndexType;
  using ScalarType = typename Neighbor_::ScalarType;

  //! \private
  inline SearchRadius(ScalarType const radius, std::vector<NeighborType>& n)
      : radius_{radius}, n_{n} {
    n_.clear();
  }

  //! \brief Visit current point.
  inline void operator()(IndexType const idx, ScalarType const dst) const {
    n_.push_back({idx, dst});
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void Sort() const { std::sort(n_.begin(), n_.end()); }

  //! \brief Maximum search distance with respect to the query point.
  inline ScalarType max() const { return radius_; }

 private:
  ScalarType radius_;
  std::vector<NeighborType>& n_;
};

//! \brief Search visitor for finding approximate nearest neighbors.
//! \details Points and tree nodes are skipped by scaling down the search
//! distance, possibly not visiting the true nearest neighbor. An approximate
//! nearest neighbor will at most be a factor of distance ratio \p e farther
//! from the query point than the true nearest neighbor: max_ann_distance =
//! true_nn_distance * e.
//!
//! There are different possible implementations to get an approximate nearest
//! neighbor but this one is (probably) the cheapest by skipping both points
//! inside leafs and complete tree nodes. Even though all points are checked
//! inside a leaf, not all of them are visited. This saves on scaling and heap
//! updates.
//! \see SearchKnn
template <typename RandomAccessIterator_>
class SearchAknn {
 public:
  static_assert(
      std::is_base_of_v<
          std::random_access_iterator_tag,
          typename std::iterator_traits<
              RandomAccessIterator_>::iterator_category>,
      "SEARCH_AKNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using NeighborType =
      typename std::iterator_traits<RandomAccessIterator_>::value_type;
  using IndexType = typename NeighborType::IndexType;
  using ScalarType = typename NeighborType::ScalarType;

  //! \private
  inline SearchAknn(
      ScalarType const e,
      RandomAccessIterator_ begin,
      RandomAccessIterator_ end)
      : re_{ScalarType(1.0) / e}, begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<ScalarType>::max();
  }

  //! \brief Visit current point.
  inline void operator()(IndexType const idx, ScalarType const dst) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    // Replace the current maximum for which the distance is scaled to be:
    // d = d / e.
    InsertSorted(begin_, active_end_, NeighborType{idx, dst * re_});
  }

  //! \brief Maximum search distance with respect to the query point.
  inline ScalarType max() const { return std::prev(end_)->distance; }

 private:
  ScalarType re_;
  RandomAccessIterator_ begin_;
  RandomAccessIterator_ end_;
  RandomAccessIterator_ active_end_;
};

}  // namespace internal

}  // namespace pico_tree
