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

  //! \brief Fills the sequence with value \p v.
  inline void Fill(std::size_t const, Scalar const v) {
    // The first argument is the size s. It should be the same as Dim.
    sequence_.fill(v);
  }

  //! \brief Returns a const reference to the current object.
  inline MoveReturnType Move() const noexcept { return *this; }

  //! \brief Returns the size of the sequence.
  inline constexpr std::size_t size() const noexcept {
    return sequence_.size();
  }

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
  //! \brief Storage.
  std::vector<Scalar> sequence_;
};

}  // namespace internal

}  // namespace pico_tree
