#pragma once

#include "map.hpp"
#include "space_traits.hpp"

namespace pico_tree {

template <typename Scalar_, Size Dim_>
struct PointTraits<PointMap<Scalar_, Dim_>> {
  using PointType = PointMap<Scalar_, Dim_>;
  using ScalarType = typename PointType::ScalarType;
  using SizeType = typename PointType::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline static ScalarType const* Coords(PointType const& point) {
    return point.data();
  }

  inline static SizeType Sdim(PointType const& point) { return point.size(); }
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

  inline static SizeType Sdim(SpaceType const& space) { return space.sdim(); }

  inline static SizeType Npts(SpaceType const& space) { return space.size(); }

  template <typename Index_>
  inline static auto PointAt(SpaceType const& space, Index_ idx) {
    return space[static_cast<SizeType>(idx)];
  }
};

}  // namespace pico_tree
