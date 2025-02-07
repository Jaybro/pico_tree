#pragma once

#include <vector>

namespace pico_tree::internal {

//! \brief Static MemoryBuffer using a vector. It is a simple memory buffer
//! making deletions of recursive elements a bit easier.
//! \details The buffer owns all memory returned by allocate() and all memory is
//! released when the buffer is destroyed.
template <typename T_>
class static_buffer {
 public:
  //! \brief Type allocated and stored by the buffer.
  using value_type = T_;

  //! Creates a static_buffer having space for \p size elements.
  inline explicit static_buffer(std::size_t const size) {
    buffer_.reserve(size);
  }

  //! \brief Creates an item and returns a pointer to it.
  template <typename... Args_>
  inline T_* allocate(Args_&&... args) {
    buffer_.emplace_back(std::forward<Args_>(args)...);
    return &buffer_.back();
  }

 private:
  //! \private
  std::vector<T_> buffer_;
};

}  // namespace pico_tree::internal
