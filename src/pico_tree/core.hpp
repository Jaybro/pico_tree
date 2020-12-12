#pragma once

//! \mainpage PicoTree is a small C++ header only library for range searches and
//! nearest neighbor searches using a KdTree.
//! \file core.hpp
//! \brief Contains various common utilities.

#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <vector>

namespace pico_tree {

//! \brief This value can be used in any template argument that wants to know
//! the spatial dimension of the search problem when it can only be known at
//! run-time. In this case the dimension of the problem is provided by the point
//! adaptor.
static constexpr int kDynamicDim = -1;

namespace internal {

//! \brief Restores the heap property of the range defined by \p begin and \p
//! end, assuming only the top element could have broken it.
//! \details Worst case performs O(2 log n) comparisons and O(log n) copies.
//! Performance will be better in practice because it is possible to "early
//! out" as soon as the first node is encountered that adheres to the heap
//! property.
//! <p/>
//! A function for replacing the top of the heap is not available using the
//! C++ stl. It is possible to use something like std::make_heap() or
//! range.push_back() followed by a std::pop_heap() to get the desired effect.
//! However, these solutions are either forced to traverse the entire range or
//! introduce overhead, which seem to make them quite a bit slower.
//! <p/>
//! It should be possible to use this function in combination with any of the
//! C++ stl heap related functions, as the documention of those suggest that the
//! heap is implemented as a binary heap.
template <typename RandomAccessIterator, typename Compare>
inline void ReplaceFrontHeap(
    RandomAccessIterator begin, RandomAccessIterator end, Compare comp) {
  auto const size = end - begin;

  if (size < 2) {
    return;
  }

  typename std::iterator_traits<RandomAccessIterator>::difference_type parent =
      0;
  auto front = std::move(begin[parent]);
  auto const last_parent = (size - 2) / 2;
  while (parent < last_parent) {
    auto child = 2 * parent + 1;
    if (comp(begin[child], begin[child + 1])) {
      ++child;
    }
    if (!comp(front, begin[child])) {
      begin[parent] = std::move(front);
      return;
    } else {
      begin[parent] = std::move(begin[child]);
      parent = child;
    }
  }
  // Everything below (but for replacing the last child) is the same as
  // inside the loop. The only difference is that for even sized vectors we
  // can't compare with the second child of the last parent.
  //
  // Assuming doing the check once outside of the loop is better?
  if (parent == last_parent) {
    auto child = 2 * parent + 1;
    if ((size & 1) == 1 && comp(begin[child], begin[child + 1])) {
      ++child;
    }
    if (comp(front, begin[child])) {
      begin[parent] = std::move(begin[child]);
      parent = child;
    }
  }
  // Last child gets replaced.
  begin[parent] = std::move(front);
}

//! \brief Inserts \p item in O(n) time at the index for which \p comp
//! first holds true. The sequence must be sorted and remains sorted after
//! insertion. The last item in the sequence is "pushed out".
//! \details The contents of the indices at which \p comp holds true are moved
//! to the next index. Thus, starting from the end of the sequence, each item[i]
//! gets replaced by item[i - 1] until \p comp results in false. The worst case
//! has n comparisons and n copies, traversing the entire sequence.
template <typename RandomAccessIterator, typename Compare>
inline void InsertSorted(
    RandomAccessIterator begin,
    RandomAccessIterator end,
    typename std::iterator_traits<RandomAccessIterator>::value_type item,
    Compare comp) {
  auto it = std::prev(end);
  for (; it > begin; --it) {
    if (comp(item, *std::prev(it))) {
      *it = std::move(*std::prev(it));
    } else {
      break;
    }
  }
  // We update the inserted element outside of the loop. This is done for the
  // case where we didn't break, simply reaching the end of the loop. This
  // happens when we need to replace the first element in the sequence (the last
  // item encountered) and were unable to reach the "else" clause.
  *it = std::move(item);
}

//! \brief Compile time dimension count handling.
template <int Dim_>
struct Dimension {
  //! \brief Returns the compile time dimension.
  inline static constexpr int Dim(int) { return Dim_; }
};

//! \brief Run time dimension count handling.
template <>
struct Dimension<kDynamicDim> {
  //! \brief Returns the run time dimension.
  inline static int Dim(int dim) { return dim; }
};

//! \brief Compile time sequence. A lot faster than the run time version.
template <typename Scalar, int Dim_>
class Sequence {
 public:
  //! \brief Return type of the Move() member function.
  //! \details An std::array is movable, which is useful if its contents are
  //! also movable. But because we store Scalars (float or double) the move
  //! results in a copy. In some cases we can prevent an unwanted copy.
  using MoveReturnType = Sequence const&;

  //! \details Access data contained in the Sequence.
  inline Scalar& operator[](std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator[](std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(std::size_t const, Scalar const v) { sequence_.fill(v); }

  //! \brief Returns a const reference to the current object.
  inline MoveReturnType Move() const noexcept { return *this; }

  //! \brief Returns the size of the sequence.
  inline constexpr std::size_t size() const noexcept {
    return sequence_.size();
  }

 private:
  //! Storage.
  std::array<Scalar, Dim_> sequence_;
};

//! \brief Run time sequence. More flexible than the compile time one.
template <typename Scalar>
class Sequence<Scalar, kDynamicDim> {
 public:
  //! \brief Return type of the Move() member function.
  //! \details Moving a vector is quite a bit cheaper than copying it. The
  //! std::array version of Sequence cannot be moved and this return type allows
  //! the using code to be agnostic to the actual storage type.
  using MoveReturnType = Sequence&&;

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator[](std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator[](std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Changes the size of the sequence to \p s and fills the sequence
  //! with value \p v.
  inline void Fill(std::size_t const s, Scalar const v) {
    sequence_.assign(s, v);
  }

  //! \brief Moves the current object.
  inline MoveReturnType Move() noexcept { return std::move(*this); }

  //! \brief Returns the size of the sequence.
  inline constexpr std::size_t size() const noexcept {
    return sequence_.size();
  }

 private:
  //! Storage.
  std::vector<Scalar> sequence_;
};

//! \brief Simple memory buffer making deletions of recursive elements a bit
//! easier.
//! \details The buffer owns all memory returned by MakeItem() and all memory is
//! released when the buffer is destroyed.
template <typename Container>
class MemoryBuffer {
 public:
  //! Creates an item and returns a pointer to it. The buffer retains ownership
  //! of the memory. All memory gets released when the buffer is destroyed.
  template <typename... Args>
  inline typename Container::value_type* MakeItem(Args&&... args) {
    buffer_.emplace_back(std::forward<Args>(args)...);
    return &buffer_.back();
  }

 protected:
  //! Memory buffer.
  Container buffer_;
};

//! \brief Static MemoryBuffer using a vector.
//! \details The buffer owns all memory returned by MakeItem() and all memory is
//! released when the buffer is destroyed.
template <typename T>
class StaticBuffer : public MemoryBuffer<std::vector<T>> {
 public:
  //! Creates a StaticBuffer having space for \p size elements.
  inline StaticBuffer(std::size_t const size) {
    MemoryBuffer<std::vector<T>>::buffer_.reserve(size);
  }
};

//! \brief Dynamic MemoryBuffer using a deque.
//! \details The buffer owns all memory returned by MakeItem() and all memory is
//! released when the buffer is destroyed.
template <typename T>
class DynamicBuffer : public MemoryBuffer<std::deque<T>> {
 public:
  //! Creates a DynamicBuffer.
  inline DynamicBuffer() = default;
  //! Creates a DynamicBuffer. Ignores the argument in favor of a common
  //! interface with the StaticBuffer.
  inline DynamicBuffer(std::size_t const) {}
};

//! \brief Returns an std::fstream given a filename.
//! \details Convenience function that throws an std::runtime_error in case it
//! is unable to open the stream.
inline std::fstream OpenStream(
    std::string const& filename, std::ios_base::openmode mode) {
  std::fstream stream(filename, mode);

  if (!stream.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  return stream;
}

//! \brief The Stream class is an std::iostream wrapper that helps read and
//! write various simple data types.
class Stream {
 public:
  //! \brief Constructs a Stream using an input std::iostream.
  Stream(std::iostream* stream) : stream_(*stream) {}

  //! \brief Reads a single value from the stream.
  //! \tparam T Type of the value.
  template <typename T>
  inline void Read(T* value) {
    stream_.read(reinterpret_cast<char*>(value), sizeof(T));
  }

  //! \brief Reads a vector of values from the stream.
  //! \details Reads the size of the vector followed by all its elements.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Read(std::vector<T>* values) {
    decltype(std::vector<T>().template size()) size;
    Read(&size);
    values->resize(size);
    stream_.read(reinterpret_cast<char*>(&(*values)[0]), sizeof(T) * size);
  }

  //! \brief Writes a single value to the stream.
  //! \tparam T Type of the value.
  template <typename T>
  inline void Write(T const& value) {
    stream_.write(reinterpret_cast<char const*>(&value), sizeof(T));
  }

  //! \brief Writes a vector of values to the stream.
  //! \details Writes the size of the vector followed by all its elements.
  //! \tparam T Type of a value.
  template <typename T>
  inline void Write(std::vector<T> const& values) {
    Write(values.size());
    stream_.write(
        reinterpret_cast<char const*>(&values[0]), sizeof(T) * values.size());
  }

 private:
  //! \brief Wrapped stream.
  std::iostream& stream_;
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
