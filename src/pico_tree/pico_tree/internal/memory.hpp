#pragma once

#include <type_traits>

namespace pico_tree {

namespace internal {

//! \brief A ListPool is useful for creating objects of a single type when the
//! total amount to be created cannot be known up front. The list maintains
//! ownership of the objects which are destructed when the list is destructed.
//! Objects cannot be returned to the pool.
//! \details A ListPool maintains a linked list of fixed size chunks that
//! contain a single object type. The size of the list increases as more objects
//! are requested.
//! <p/>
//! This class supersedes the use of the std::deque. The std::deque appears to
//! to have different a chunk size depending on the implementation of the C++
//! standard. This practically means that the performance of PicoTree is
//! unstable across platforms.
//! <p/>
//! Benchmarked various vs. the ListPool using a chunk size 256:
//! * GCC libstdc++ std::deque ~50% slower.
//! * Other variations (like the std::list) were about 10% slower.
//! <p/>
//! https://en.wikipedia.org/wiki/Memory_pool
template <typename T, std::size_t ChunkSize>
class ListPool {
 private:
  static_assert(std::is_trivial<T>::value, "TYPE_T_IS_NOT_TRIVIAL");
  static_assert(
      std::is_trivially_destructible<T>::value,
      "TYPE_T_IS_NOT_TRIVIALLY_DESTRUCTIBLE");
  //! \brief List item.
  struct Chunk {
    Chunk* prev;
    T data[ChunkSize];
  };

 public:
  //! \brief Type allocated and stored by the ListPool.
  using ValueType = T;

  //! \brief Creates a ListPool using the default constructor.
  ListPool() : end_(nullptr), index_(ChunkSize) {}

  //! \brief A ListPool instance cannot be copied.
  //! \details The default copy constructor would just copy the pointer owned by
  //! the ListPool. Also, a ListPool provides pointers to objects. Creating new
  //! objects would invalidate those pointers.
  ListPool(ListPool const&) = delete;

  //! \brief ListPool move constructor.
  ListPool(ListPool&& other) {
    end_ = other.end_;
    index_ = other.index_;
    // So we don't accidentally delete things twice.
    other.end_ = nullptr;
  }

  //! \brief A ListPool instance cannot be copied.
  ListPool& operator=(ListPool const& other) = delete;

  //! \brief ListPool move assignment.
  ListPool& operator=(ListPool&& other) {
    end_ = other.end_;
    index_ = other.index_;
    other.end_ = nullptr;
    return *this;
  }

  //! \brief Destroys up the ListPool using the destructor.
  ~ListPool() {
    // Suppose Chunk was contained by an std::unique_ptr, then it may happen
    // that we hit a recursion limit depending on how many chunks are
    // destructed.
    while (end_ != nullptr) {
      Chunk* chunk = end_->prev;
      delete end_;
      end_ = chunk;
    }
  }

  //! \brief Creates an item and returns a pointer to it.
  inline T* Allocate() {
    if (index_ == ChunkSize) {
      Chunk* chunk = new Chunk;
      chunk->prev = end_;
      end_ = chunk;
      index_ = 0;
    }

    T* i = &end_->data[index_];
    index_++;

    return i;
  }

 private:
  //! \brief The last and currently active chunk.
  Chunk* end_;
  //! \brief Index within the last chunk.
  std::size_t index_;
};

}  // namespace internal

}  // namespace pico_tree
