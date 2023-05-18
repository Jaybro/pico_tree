#pragma once

#include <algorithm>
#include <array>

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

//! \details The non-specialized class knows its dimension at compile-time and
//! uses an std::array for storing its data. Faster than using the std::vector
//! in practice.
template <typename Scalar_, Size Dim_>
struct PointStorage {
  constexpr explicit PointStorage(Size) {}

  std::array<Scalar_, Dim_> container;
};

//! \details The specialized class doesn't knows its dimension at compile-time
//! and uses an std::vector for storing its data so it can be resized.
template <typename Scalar_>
struct PointStorage<Scalar_, kDynamicSize> {
  constexpr explicit PointStorage(Size size) : container(size) {}

  std::vector<Scalar_> container;
};

//! \brief A sequence stores a contiguous array of elements similar to an
//! std::array or std::vector.
template <typename Scalar_, Size Dim_>
class Point {
 public:
  static_assert(Dim_ == kDynamicSize || Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;

  constexpr Point() : storage_(Dim_) {}

  constexpr explicit Point(SizeType size) : storage_(size) {}

  //! \brief Fills the storage with value \p v.
  constexpr void Fill(ScalarType v) {
    std::fill(storage_.container.begin(), storage_.container.end(), v);
  }

  //! \brief Access the container data.
  constexpr ScalarType& operator[](SizeType i) noexcept {
    return storage_.container[i];
  }

  //! \brief Access the container data.
  constexpr ScalarType const& operator[](SizeType i) const noexcept {
    return storage_.container[i];
  }

  constexpr ScalarType const* data() const noexcept {
    return storage_.container.data();
  }

  constexpr ScalarType* data() noexcept { return storage_.container.data(); }

  //! \brief Returns the size of the container.
  constexpr SizeType size() const noexcept { return storage_.container.size(); }

 private:
  PointStorage<Scalar_, Dim_> storage_;
};

}  // namespace internal

}  // namespace pico_tree
