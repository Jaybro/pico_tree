#pragma once

#include <vector>

namespace nano_tree {

//! Value used for any template Dims parameter to determine the dimensions
//! compile time.
constexpr int kRuntimeDims = -1;

namespace internal {

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