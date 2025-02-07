#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/point_traits.hpp"

namespace pico_tree {

template <typename Scalar_, std::size_t Dim_>
struct point_traits<internal::point<Scalar_, Dim_>> {
  using point_type = internal::point<Scalar_, Dim_>;
  using scalar_type = Scalar_;
  static size_t constexpr dim = Dim_;

  inline static scalar_type const* data(point_type const& point) {
    return point.data();
  }

  inline static std::size_t constexpr size(point_type const& point) {
    return point.size();
  }
};

}  // namespace pico_tree
