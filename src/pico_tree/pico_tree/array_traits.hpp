#pragma once

#include <array>

#include "point_traits.hpp"

namespace pico_tree {

//! \brief Point interface for Scalar_[Dim_].
template <typename Scalar_, std::size_t Dim_>
struct PointTraits<Scalar_[Dim_]> {
  using PointType = Scalar_[Dim_];
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  //! \brief Returns a pointer to the coordinates of the input point.
  inline static constexpr ScalarType const* data(PointType const& point) {
    return point;
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static constexpr std::size_t size(PointType const&) { return Dim; }
};

//! \brief Point interface for std::array<Scalar_, Dim_>.
template <typename Scalar_, std::size_t Dim_>
struct PointTraits<std::array<Scalar_, Dim_>> {
  using PointType = std::array<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  //! \brief Returns a pointer to the coordinates of the input point.
  inline static constexpr ScalarType const* data(PointType const& point) {
    return point.data();
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static constexpr std::size_t size(PointType const& point) {
    return point.size();
  }
};

}  // namespace pico_tree
