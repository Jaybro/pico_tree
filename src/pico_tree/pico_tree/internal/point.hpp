#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include "pico_tree/core.hpp"

namespace pico_tree::internal {

//! \details The non-specialized class knows its dimension at compile-time and
//! uses an std::array for storing its data. Faster than using the std::vector
//! in practice.
template <typename Scalar_, Size Dim_>
struct PointStorageTraits {
  using Type = std::array<Scalar_, Dim_>;

  static constexpr auto FromSize([[maybe_unused]] Size size) {
    assert(size == Dim_);
    return Type();
  }
};

//! \details The specialized class doesn't knows its dimension at compile-time
//! and uses an std::vector for storing its data so it can be resized.
template <typename Scalar_>
struct PointStorageTraits<Scalar_, kDynamicSize> {
  using Type = std::vector<Scalar_>;

  static constexpr auto FromSize(Size size) { return Type(size); }
};

//! \brief A Point is a container that stores a contiguous array of elements as
//! an aggregate type. The storage is either as an std::array or an std::vector.
//! Using the storage, elems_, is considered undefined behavior.
//! \details Having elems_ public goes against the against the encapsulation
//! principle but gives us aggregate initialization in return.
template <typename Scalar_, Size Dim_>
struct Point {
  static_assert(Dim_ == kDynamicSize || Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  //! \private Using Pst__ is considered undefined behavior.
  using Pst__ = PointStorageTraits<Scalar_, Dim_>;
  //! \private Using Elems__ is considered undefined behavior.
  using Elems__ = typename Pst__::Type;

  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;

  //! \brief Creates a point from a size.
  static constexpr Point FromSize(Size size) { return {Pst__::FromSize(size)}; }

  //! \brief Fills the storage with value \p v.
  constexpr void Fill(ScalarType v) {
    std::fill(elems_.begin(), elems_.end(), v);
  }

  //! \brief Access the container data.
  constexpr ScalarType& operator[](SizeType i) noexcept { return elems_[i]; }

  //! \brief Access the container data.
  constexpr ScalarType const& operator[](SizeType i) const noexcept {
    return elems_[i];
  }

  constexpr ScalarType const* data() const noexcept { return elems_.data(); }

  constexpr ScalarType* data() noexcept { return elems_.data(); }

  //! \brief Returns the size of the container.
  constexpr SizeType size() const noexcept { return elems_.size(); }

  //! \private Using elems_ is considered undefined behavior.
  Elems__ elems_;
};

}  // namespace pico_tree::internal
