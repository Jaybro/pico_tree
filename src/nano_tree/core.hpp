#pragma once

#include <cmath>
#include <vector>

namespace nano_tree {

//! Value used for any template Dims parameter to determine the dimensions
//! compile time.
constexpr int kRuntimeDims = -1;

namespace internal {

//! Compile time dimension count handling.
template <int Dims_>
struct Dimensions {
  //! Returns the dimension index of the dim dimension from the back.
  inline static constexpr int Back(int dim) { return Dims_ - dim; }
  inline static constexpr int Dims(int) { return Dims_; }
};

//! Run time dimension count handling.
template <>
struct Dimensions<kRuntimeDims> {
  inline static constexpr int Back(int) { return kRuntimeDims; }
  inline static int Dims(int dims) { return dims; }
};

//! Simple memory buffer making deletions of recursive elements a bit easier.
template <typename T>
class ItemBuffer {
 public:
  ItemBuffer(std::size_t const size) { buffer_.reserve(size); }

  template <typename... Args>
  inline T* MakeItem(Args&&... args) {
    buffer_.emplace_back(std::forward<Args>(args)...);
    return &buffer_.back();
  }

 private:
  std::vector<T> buffer_;
};

//! Returns the maximum amount of leaves given \p num_points and \p
//! max_leaf_size.
inline std::size_t MaxLeavesFromPoints(
    std::size_t const num_points, std::size_t const max_leaf_size) {
  // Each increase of the leaf size by a factor of two reduces the tree height
  // by 1. Each reduction in tree height halves the amount of leaves.
  // Rounding up the number of leaves means that the last one is not fully
  // occupied.
  return std::ceil(
      num_points /
      std::pow(2.0, std::floor(std::log2(static_cast<double>(max_leaf_size)))));
}

//! Returns the maximum amount of nodes given \p num_points and \p
//! max_leaf_size.
inline std::size_t MaxNodesFromPoints(
    std::size_t const num_points, std::size_t const max_leaf_size = 1) {
  return MaxLeavesFromPoints(num_points, max_leaf_size) * 2 - 1;
}

}  // namespace internal

}  // namespace nano_tree
