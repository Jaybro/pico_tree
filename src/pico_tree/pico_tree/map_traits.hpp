#pragma once

//! \file map_traits.hpp
//! \brief Provides an interface for spaces and points when working with raw
//! pointers.

#include "map.hpp"
#include "space_traits.hpp"

namespace pico_tree {

template <typename Scalar_, Size Dim_>
struct PointTraits<PointMap<Scalar_, Dim_>> {
  using PointType = PointMap<Scalar_, Dim_>;
  using ScalarType = typename PointType::ScalarType;
  using SizeType = typename PointType::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline static ScalarType const* data(PointType const& point) {
    return point.data();
  }

  inline static SizeType size(PointType const& point) { return point.size(); }
};

//! \brief MapTraits provides an interface for spaces and points when working
//! with a SpaceMap.
template <typename Point_>
struct SpaceTraits<SpaceMap<Point_>> {
  using SpaceType = SpaceMap<Point_>;
  using PointType = typename SpaceType::PointType;
  using ScalarType = typename SpaceType::ScalarType;
  using SizeType = typename SpaceType::SizeType;
  static SizeType constexpr Dim = SpaceType::Dim;

  template <typename Index_>
  inline static decltype(auto) PointAt(SpaceType const& space, Index_ idx) {
    return space[static_cast<SizeType>(idx)];
  }

  inline static SizeType size(SpaceType const& space) { return space.size(); }

  inline static SizeType sdim(SpaceType const& space) { return space.sdim(); }
};

}  // namespace pico_tree
