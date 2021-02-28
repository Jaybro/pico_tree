#pragma once

#include <algorithm>

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

//! \brief KdTree search visitor for finding a single nearest neighbor.
template <typename Neighbor>
class SearchNn {
 private:
  using Index = typename Neighbor::IndexType;
  using Scalar = typename Neighbor::ScalarType;

 public:
  //! \private
  inline SearchNn(Neighbor* nn) : nn_{*nn} {
    nn_.distance = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const dst) const {
    nn_ = {idx, dst};
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return nn_.distance; }

 private:
  Neighbor& nn_;
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
template <typename RandomAccessIterator>
class SearchKnn {
 private:
  static_assert(
      std::is_same<
          typename std::iterator_traits<
              RandomAccessIterator>::iterator_category,
          std::random_access_iterator_tag>::value,
      "SEARCH_KNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using Neighbor =
      typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Index = typename Neighbor::IndexType;
  using Scalar = typename Neighbor::ScalarType;

 public:
  //! \private
  inline SearchKnn(RandomAccessIterator begin, RandomAccessIterator end)
      : begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const dst) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    InsertSorted(begin_, active_end_, Neighbor{idx, dst});
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return std::prev(end_)->distance; }

 private:
  RandomAccessIterator begin_;
  RandomAccessIterator end_;
  RandomAccessIterator active_end_;
};

//! \brief KdTree search visitor for finding all neighbors within a radius.
template <typename Neighbor>
class SearchRadius {
 private:
  using Index = typename Neighbor::IndexType;
  using Scalar = typename Neighbor::ScalarType;

 public:
  //! \private
  inline SearchRadius(Scalar const radius, std::vector<Neighbor>* n)
      : radius_{radius}, n_{*n} {
    n_.clear();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const dst) const {
    n_.push_back({idx, dst});
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void Sort() const { std::sort(n_.begin(), n_.end()); }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return radius_; }

 private:
  Scalar const radius_;
  std::vector<Neighbor>& n_;
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
template <typename RandomAccessIterator>
class SearchAknn {
 private:
  static_assert(
      std::is_same<
          typename std::iterator_traits<
              RandomAccessIterator>::iterator_category,
          std::random_access_iterator_tag>::value,
      "SEARCH_AKNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using Neighbor =
      typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Index = typename Neighbor::IndexType;
  using Scalar = typename Neighbor::ScalarType;

 public:
  //! \private
  inline SearchAknn(
      Scalar const e, RandomAccessIterator begin, RandomAccessIterator end)
      : re_{Scalar(1.0) / e}, begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->distance = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const dst) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    // Replace the current maximum for which the distance is scaled to be:
    // d = d / e.
    InsertSorted(begin_, active_end_, Neighbor{idx, dst * re_});
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return std::prev(end_)->distance; }

 private:
  Scalar re_;
  RandomAccessIterator begin_;
  RandomAccessIterator end_;
  RandomAccessIterator active_end_;
};

}  // namespace internal

}  // namespace pico_tree
