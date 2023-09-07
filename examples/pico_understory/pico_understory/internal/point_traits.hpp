#pragma once

#include "pico_tree/internal/point.hpp"
#include "pico_tree/point_traits.hpp"

namespace pico_tree {

template <typename Scalar_, std::size_t Dim_>
struct PointTraits<internal::Point<Scalar_, Dim_>> {
  using PointType = internal::Point<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  inline static ScalarType const* data(PointType const& point) {
    return point.data();
  }

  inline static std::size_t constexpr size(PointType const& point) {
    return point.size();
  }
};

}  // namespace pico_tree
