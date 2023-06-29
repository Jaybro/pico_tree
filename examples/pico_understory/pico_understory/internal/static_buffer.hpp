#pragma once

#include <vector>

namespace pico_tree::internal {

//! \brief Static MemoryBuffer using a vector. It is a simple memory buffer
//! making deletions of recursive elements a bit easier.
//! \details The buffer owns all memory returned by Allocate() and all memory is
//! released when the buffer is destroyed.
template <typename T>
class StaticBuffer {
 public:
  //! \brief Type allocated and stored by the buffer.
  using ValueType = T;

  //! Creates a StaticBuffer having space for \p size elements.
  inline explicit StaticBuffer(std::size_t const size) {
    buffer_.reserve(size);
  }

  //! \brief Creates an item and returns a pointer to it.
  template <typename... Args>
  inline T* Allocate(Args&&... args) {
    buffer_.emplace_back(std::forward<Args>(args)...);
    return &buffer_.back();
  }

 private:
  //! \private
  std::vector<T> buffer_;
};

}  // namespace pico_tree::internal
