#pragma once

//! \mainpage PicoTree is a small C++ header only library for range searches and
//! nearest neighbor searches using a KdTree.
//! \file core.hpp
//! \brief Contains various common utilities.

#include <array>
#include <cassert>
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

//! \brief A Neighbor is a point reference with a corresponding distance to
//! another point.
template <typename Index_, typename Scalar_>
struct Neighbor {
  static_assert(std::is_integral<Index_>::value);
  static_assert(
      std::is_integral<Scalar_>::value ||
      std::is_floating_point<Scalar_>::value);

  //! \brief Index type.
  using Index = Index_;
  //! \brief Distance type.
  using Scalar = Scalar_;

  //! \brief Default constructor.
  //! \details Declaring a custom constructor removes the default one. With
  //! C++11 we can bring back the default constructor and keep this struct a POD
  //! type.
  inline constexpr Neighbor() = default;
  //! \brief Constructs a Neighbor given an index and distance.
  inline constexpr Neighbor(Index idx, Scalar dst) noexcept
      : index(std::forward<Index>(idx)), distance(std::forward<Scalar>(dst)) {}

  //! \brief Point index of the Neighbor.
  Index index;
  //! \brief Distance of the Neighbor with respect to another point.
  Scalar distance;
};

//! \brief Compares neighbors by distance.
template <typename Index, typename Scalar>
inline constexpr bool operator<(
    Neighbor<Index, Scalar> const& lhs, Neighbor<Index, Scalar> const& rhs) {
  return lhs.distance < rhs.distance;
}

namespace internal {

//! \brief Inserts \p item in O(n) time at the index for which \p comp
//! first holds true. The sequence must be sorted and remains sorted after
//! insertion. The last item in the sequence is "pushed out".
//! \details The contents of the indices at which \p comp holds true are moved
//! to the next index. Thus, starting from the end of the sequence, each item[i]
//! gets replaced by item[i - 1] until \p comp results in false. The worst case
//! has n comparisons and n copies, traversing the entire sequence.
//! <p/>
//! This algorithm is used as the inner loop of insertion sort:
//! * https://en.wikipedia.org/wiki/Insertion_sort
template <
    typename RandomAccessIterator,
    typename Compare = std::less<
        typename std::iterator_traits<RandomAccessIterator>::value_type>>
inline void InsertSorted(
    RandomAccessIterator begin,
    RandomAccessIterator end,
    typename std::iterator_traits<RandomAccessIterator>::value_type item,
    Compare comp = Compare()) {
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

//! \brief A sequence stores a contiguous array of elements similar to
//! std::array or std::vector.
//! \details The non-specialized Sequence class knows its dimension at
//! compile-time and uses an std::array for storing its data. Faster than using
//! the std::vector in practice.
template <typename Scalar, int Dim_>
class Sequence {
 private:
  static_assert(Dim_ >= 0);

 public:
  //! \brief Return type of the Move() member function.
  //! \details An std::array is movable, which is useful if its contents are
  //! also movable. But because we store Scalars (float or double) the move
  //! results in a copy. In some cases we can prevent an unwanted copy.
  using MoveReturnType = Sequence const&;

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator[](std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator[](std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator()(std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator()(std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(std::size_t const s, Scalar const v) {
    assert(s == static_cast<std::size_t>(Dim_));
    sequence_.fill(v);
  }

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

//! \brief A sequence stores a contiguous array of elements similar to
//! std::array or std::vector.
//! \details The specialized Sequence class doesn't knows its dimension at
//! compile-time and uses an std::vector for storing its data so it can be
//! resized.
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

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator()(std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator()(std::size_t const i) const noexcept {
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
