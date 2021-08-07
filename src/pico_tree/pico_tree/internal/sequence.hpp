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
template <typename Scalar, int Dim_>
class Sequence {
 private:
  static_assert(Dim_ >= 0, "SEQUENCE_DIM_MUST_BE_DYNAMIC_OR_>=_0");

 public:
  inline explicit Sequence(std::size_t const) {}

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator[](std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator[](std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(Scalar const v) { sequence_.fill(v); }

  //! \brief Returns the size of the sequence.
  inline constexpr std::size_t size() const noexcept {
    return sequence_.size();
  }

  //! \brief Returns a const reference to the underlying container.
  inline constexpr std::array<Scalar, Dim_> const& container() const noexcept {
    return sequence_;
  }

  //! \brief Returns a reference to the underlying container.
  inline std::array<Scalar, Dim_>& container() noexcept { return sequence_; }

 private:
  //! \brief Storage.
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
  inline explicit Sequence(std::size_t const size) : sequence_(size) {}

  //! \brief Access data contained in the Sequence.
  inline Scalar& operator[](std::size_t const i) noexcept {
    return sequence_[i];
  }

  //! \brief Access data contained in the Sequence.
  inline Scalar const& operator[](std::size_t const i) const noexcept {
    return sequence_[i];
  }

  //! \brief Fills the sequence with value \p v.
  inline void Fill(Scalar const v) { sequence_.assign(sequence_.size(), v); }

  //! \brief Returns the size of the sequence.
  inline constexpr std::size_t size() const noexcept {
    return sequence_.size();
  }

  //! \brief Returns a const reference to the underlying container.
  inline std::vector<Scalar> const& container() const noexcept {
    return sequence_;
  }

  //! \brief Returns a reference to the underlying container.
  inline std::vector<Scalar>& container() noexcept { return sequence_; }

 private:
  //! \brief Storage.
  std::vector<Scalar> sequence_;
};

}  // namespace internal

}  // namespace pico_tree
