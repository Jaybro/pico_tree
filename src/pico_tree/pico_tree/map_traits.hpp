#pragma once

#include "map.hpp"

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
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Point_, typename Index_ = int>
struct MapTraits {
  using SpaceType = SpaceMap<Point_>;
  using PointType = typename SpaceType::PointType;
  using ScalarType = typename SpaceType::ScalarType;
  using SizeType = typename SpaceType::SizeType;
  static SizeType constexpr Dim = SpaceType::Dim;
  using IndexType = Index_;

  //! \brief Returns the traits for the given input point type.
  template <typename OtherPoint_>
  using PointTraitsFor = PointTraits<OtherPoint_>;

  inline static SizeType Sdim(SpaceType const& space) { return space.sdim(); }

  inline static IndexType Npts(SpaceType const& space) {
    return static_cast<IndexType>(space.size());
  }

  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space[idx];
  }
};

}  // namespace pico_tree
