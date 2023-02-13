#pragma once

#include "map.hpp"

namespace pico_tree {

template <typename Scalar_, Size Dim_>
struct StdPointTraits<PointMap<Scalar_, Dim_>> {
  using PointType = PointMap<Scalar_, Dim_>;
  using ScalarType = typename PointType::ScalarType;
  using SizeType = typename PointType::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline static ScalarType const* Coords(PointType const& point) {
    return point.data();
  }

  inline static SizeType Sdim(PointType const& point) { return point.size(); }
};

//! \brief MapTraits provides an interface for SpaceMap and points supported by
//! StdPointTraits.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Scalar_, Size Dim_, typename Index_ = int>
struct MapTraits {
  using PointType = PointMap<Scalar_, Dim_>;
  using SpaceType = SpaceMap<PointType>;
  using ScalarType = typename SpaceType::ScalarType;
  using SizeType = typename SpaceType::SizeType;
  static SizeType constexpr Dim = SpaceType::Dim;
  using IndexType = Index_;

  inline static SizeType SpaceSdim(SpaceType const& space) {
    return space.sdim();
  }

  inline static IndexType SpaceNpts(SpaceType const& space) {
    return static_cast<IndexType>(space.size());
  }

  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space[idx];
  }

  template <typename OtherPoint>
  inline static SizeType PointSdim(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Sdim(point);
  }

  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pico_tree
