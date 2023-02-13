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

  std::array<Scalar_, Dim_> point;
};

//! \details The specialized class doesn't knows its dimension at compile-time
//! and uses an std::vector for storing its data so it can be resized.
template <typename Scalar_>
struct PointStorage<Scalar_, kDynamicDim> {
  constexpr explicit PointStorage(Size size) : point(size) {}

  std::vector<Scalar_> point;
};

//! \brief A sequence stores a contiguous array of elements similar to an
//! std::array or std::vector.
template <typename Scalar_, Size Dim_>
class Sequence {
 public:
  static_assert(Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;

  constexpr Sequence() : storage_(Dim_) {}

  constexpr explicit Sequence(SizeType size) : storage_(size) {}

  //! \brief Fills the storage with value \p v.
  inline void Fill(ScalarType v) {
    std::fill(storage_.point.begin(), storage_.point.end(), v);
  }

  //! \brief Access the container data.
  inline ScalarType& operator[](SizeType i) noexcept {
    return storage_.point[i];
  }

  //! \brief Access the container data.
  inline ScalarType const& operator[](SizeType i) const noexcept {
    return storage_.point[i];
  }

  //! \brief Returns the size of the container.
  constexpr SizeType size() const noexcept { return storage_.point.size(); }

 private:
  PointStorage<Scalar_, Dim_> storage_;
};

}  // namespace internal

}  // namespace pico_tree
