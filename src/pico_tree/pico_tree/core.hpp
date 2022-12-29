#pragma once

//! \mainpage PicoTree is a C++ header only library for nearest neighbor
//! searches and range searches using a KdTree.
//! \file core.hpp
//! \brief Contains various common utilities.

#include <cmath>
#include <functional>
#include <vector>

namespace pico_tree {

//! \brief Size type used by PicoTree.
using Size = std::size_t;

//! \brief This value can be used in any template argument that wants to know
//! the spatial dimension of the search problem when it can only be known at
//! run-time. In this case the dimension of the problem is provided by the point
//! adaptor.
static Size constexpr kDynamicDim = -1;

//! \brief A Neighbor is a point reference with a corresponding distance to
//! another point.
template <typename Index_, typename Scalar_>
struct Neighbor {
  static_assert(std::is_integral<Index_>::value, "INDEX_NOT_AN_INTEGRAL_TYPE");
  static_assert(
      std::is_integral<Scalar_>::value ||
          std::is_floating_point<Scalar_>::value,
      "SCALAR_NOT_AN_INTEGRAL_OR_FLOATING_POINT_TYPE");

  //! \brief Index type.
  using IndexType = Index_;
  //! \brief Distance type.
  using ScalarType = Scalar_;

  //! \brief Default constructor.
  //! \details Declaring a custom constructor removes the default one. With
  //! C++11 we can bring back the default constructor and keep this struct a POD
  //! type.
  inline constexpr Neighbor() = default;
  //! \brief Constructs a Neighbor given an index and distance.
  inline constexpr Neighbor(IndexType idx, ScalarType dst) noexcept
      : index(idx), distance(dst) {}

  //! \brief Point index of the Neighbor.
  IndexType index;
  //! \brief Distance of the Neighbor with respect to another point.
  ScalarType distance;
};

//! \brief Compares neighbors by distance.
template <typename Index_, typename Scalar_>
inline constexpr bool operator<(
    Neighbor<Index_, Scalar_> const& lhs,
    Neighbor<Index_, Scalar_> const& rhs) noexcept {
  return lhs.distance < rhs.distance;
}

namespace internal {

//! \brief Inserts \p item in O(n) time at the index for which \p comp first
//! holds true. The sequence must be sorted and remains sorted after insertion.
//! The last item in the sequence is "pushed out".
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

//! \brief Compile time dimension count handling.
template <typename Traits_, Size Dim_ = Traits_::Dim>
struct Dimension {
  //! \brief Returns the compile time dimension of a space.
  inline static Size constexpr Dim(typename Traits_::SpaceType const&) {
    return Dim_;
  }

  //! \brief Returns the compile time dimension of a point.
  template <typename P>
  inline static Size constexpr Dim(P const&) {
    return Dim_;
  }
};

//! \brief Run time dimension count handling.
template <typename Traits_>
struct Dimension<Traits_, pico_tree::kDynamicDim> {
  //! \brief Returns the run time dimension of a space.
  inline static Size Dim(typename Traits_::SpaceType const& space) {
    return Traits_::SpaceSdim(space);
  }

  //! \brief Returns the run time dimension of a point.
  template <typename P>
  inline static Size Dim(P const& point) {
    return Traits_::PointSdim(point);
  }
};

//! \brief Returns the maximum amount of leaves given \p num_points and \p
//! max_leaf_size.
inline std::size_t MaxLeavesFromPoints(
    std::size_t const num_points, std::size_t const max_leaf_size) {
  // Each increase of the leaf size by a factor of two reduces the tree height
  // by 1. Each reduction in tree height halves the amount of leaves.
  // Rounding up the number of leaves means that the last one is not fully
  // occupied.
  return static_cast<std::size_t>(std::ceil(
      num_points /
      std::pow(
          2.0, std::floor(std::log2(static_cast<double>(max_leaf_size))))));
}

//! \brief Returns the maximum amount of nodes given \p num_points and \p
//! max_leaf_size.
inline std::size_t MaxNodesFromPoints(
    std::size_t const num_points, std::size_t const max_leaf_size = 1) {
  return MaxLeavesFromPoints(num_points, max_leaf_size) * 2 - 1;
}

}  // namespace internal

}  // namespace pico_tree
