#pragma once

#include "matrix_space.hpp"
#include "pico_tree/map_traits.hpp"

namespace pico_tree {

template <typename Scalar_, Size Dim_>
struct SpaceTraits<internal::MatrixSpace<Scalar_, Dim_>> {
  using SpaceType = internal::MatrixSpace<Scalar_, Dim_>;
  using PointType = PointMap<Scalar_ const, Dim_>;
  using ScalarType = typename SpaceType::ScalarType;
  using SizeType = typename SpaceType::SizeType;
  static SizeType constexpr Dim = SpaceType::Dim;

  template <typename Index_>
  inline static PointType PointAt(SpaceType const& space, Index_ idx) {
    return space[static_cast<SizeType>(idx)];
  }

  inline static SizeType size(SpaceType const& space) { return space.size(); }

  inline static SizeType sdim(SpaceType const& space) { return space.sdim(); }
};

}  // namespace pico_tree
