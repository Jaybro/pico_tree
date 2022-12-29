#pragma once

#include <array>

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

//! \brief A sequence stores a contiguous array of elements similar to
//! std::array or std::vector.
//! \details The non-specialized Sequence class knows its dimension at
//! compile-time and uses an std::array for storing its data. Faster than using
//! the std::vector in practice.
template <typename Scalar_, Size Dim_>
class Sequence {
 public:
  static_assert(Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ScalarType = Scalar_;
  using SizeType = Size;

  inline explicit Sequence(SizeType) {}

  //! \brief Access data contained in the Sequence.
  inline ScalarType& operator[](SizeType i) noexcept { return sequence_[i]; }

  //! \brief Access data contained in the Sequence.
  inline ScalarType const& operator[](SizeType i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(ScalarType v) { sequence_.fill(v); }

  //! \brief Returns the size of the sequence.
  inline constexpr SizeType size() const noexcept { return sequence_.size(); }

 private:
  //! \brief Storage.
  std::array<ScalarType, Dim_> sequence_;
};

//! \brief A sequence stores a contiguous array of elements similar to
//! std::array or std::vector.
//! \details The specialized Sequence class doesn't knows its dimension at
//! compile-time and uses an std::vector for storing its data so it can be
//! resized.
template <typename Scalar_>
class Sequence<Scalar_, kDynamicDim> {
 public:
  using ScalarType = Scalar_;
  using SizeType = Size;

  inline explicit Sequence(SizeType size) : sequence_(size) {}

  //! \brief Access data contained in the Sequence.
  inline ScalarType& operator[](SizeType i) noexcept { return sequence_[i]; }

  //! \brief Access data contained in the Sequence.
  inline ScalarType const& operator[](SizeType i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(ScalarType v) { sequence_.assign(sequence_.size(), v); }

  //! \brief Returns the size of the sequence.
  inline constexpr SizeType size() const noexcept { return sequence_.size(); }

 private:
  //! \brief Storage.
  std::vector<ScalarType> sequence_;
};

}  // namespace internal

}  // namespace pico_tree
