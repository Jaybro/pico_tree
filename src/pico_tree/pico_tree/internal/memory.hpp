#pragma once

#include <type_traits>

namespace pico_tree {

namespace internal {

//! \brief An instance of ListPoolResource constructs fixed size chunks of
//! memory and stores these in a list. Memory is only released when the resource
//! is destructed or when calling the Release() method.
//! \details A ListPoolResource is mainly useful for monotonically constructing
//! objects of a single type when the total amount to be created cannot be known
//! up front.
//! <p/>
//! A previous memory manager implementation was based on the std::deque. The
//! chunk size that it uses can vary across different implementations of the C++
//! standard, resulting in an unreliable performance of PicoTree.
//! <p/>
//! https://en.wikipedia.org/wiki/Memory_pool
template <typename T, std::size_t ChunkSize>
class ListPoolResource {
 private:
  static_assert(std::is_trivial<T>::value, "TYPE_T_IS_NOT_TRIVIAL");
  static_assert(
      std::is_trivially_destructible<T>::value,
      "TYPE_T_IS_NOT_TRIVIALLY_DESTRUCTIBLE");

  //! \brief Chunk of memory.
  struct Chunk {
    Chunk* prev;
    std::array<T, ChunkSize> data;
  };

 public:
  //! \brief Type allocated by the ListPoolResource.
  using ValueType = T;

  //! \brief ListPoolResource constructor.
  ListPoolResource() : head_(nullptr) {}

  //! \brief A ListPoolResource instance cannot be copied.
  //! \details Just no!
  ListPoolResource(ListPoolResource const&) = delete;

  //! \private
  ListPoolResource(ListPoolResource&& other) : head_(other.head_) {
    // So we don't accidentally delete things twice.
    other.head_ = nullptr;
  }

  //! \private
  ListPoolResource& operator=(ListPoolResource const& other) = delete;

  //! \private
  ListPoolResource& operator=(ListPoolResource&& other) {
    head_ = other.head_;
    other.head_ = nullptr;
    return *this;
  }

  //! \brief ListPoolResource destructor.
  virtual ~ListPoolResource() { Release(); }

  //! \brief Allocates a chunk of memory and returns a pointer to it.
  inline std::array<T, ChunkSize>* Allocate() {
    Chunk* chunk = new Chunk;
    chunk->prev = head_;
    head_ = chunk;
    return &head_->data;
  }

  //! \brief Release all memory allocated by this ListPoolResource.
  void Release() {
    // Suppose Chunk was contained by an std::unique_ptr, then it may happen
    // that we hit a recursion limit depending on how many chunks are
    // destructed.
    while (head_ != nullptr) {
      Chunk* chunk = head_->prev;
      delete head_;
      head_ = chunk;
    }
  }

 private:
  Chunk* head_;
};

//! \brief An instance of ChunkAllocator constructs objects. It does so in
//! chunks of size ChunkSize to reduce memory fragmentation.
template <typename T, std::size_t ChunkSize>
class ChunkAllocator final {
 public:
  //! \brief Type allocated by the ChunkAllocator.
  using ValueType = T;

  ChunkAllocator() : object_index_(ChunkSize) {}

  //! \brief Create an object of type T and return a pointer to it.
  inline T* Allocate() {
    if (object_index_ == ChunkSize) {
      chunk_ = resource_.Allocate();
      object_index_ = 0;
    }

    T* object = &(*chunk_)[object_index_];
    object_index_++;

    return object;
  }

 private:
  ListPoolResource<T, ChunkSize> resource_;
  std::size_t object_index_;
  std::array<T, ChunkSize>* chunk_;
};

}  // namespace internal

}  // namespace pico_tree
