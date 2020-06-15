#pragma once

#include <vector>

namespace nano_tree {

//! Value used for any template Dims parameter to determine the dimensions
//! compile time.
constexpr int kRuntimeDims = -1;

namespace internal {

//! Knowing the dimension count at compile time we can get some added
//! information.
template <int Dims_>
struct Dimensions {
  //! Returns the dimension index of the dim dimension from the back.
  inline static constexpr int Back(int dim) { return Dims_ - dim; }
  inline static constexpr int Dims(int) { return Dims_; }
};

//! At runtime we have know added information.
template <>
struct Dimensions<kRuntimeDims> {
  inline static constexpr int Back(int) { return kRuntimeDims; }
  inline static int Dims(int dims) { return dims; }
};

//! Simple memory buffer making deletions of recursive elements a bit easier.
template <typename T>
class ItemBuffer {
 public:
  ItemBuffer(std::size_t size) { buffer_.reserve(size); }

  template <typename... Args>
  inline T* MakeItem(Args&&... args) {
    buffer_.emplace_back(std::forward<Args>(args)...);
    return &buffer_.back();
  }

 private:
  std::vector<T> buffer_;
};

//! Returns the maximum amount of nodes based on the amount of input points.
inline std::size_t MaxNodesFromPoints(std::size_t num_points) {
  return num_points * 2 - 1;
}

}  // namespace internal

}  // namespace nano_tree