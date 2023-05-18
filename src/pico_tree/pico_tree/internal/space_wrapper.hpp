#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/box.hpp"

namespace pico_tree {

namespace internal {

template <typename Traits_>
class SpaceWrapper {
 public:
  using SpaceType = typename Traits_::SpaceType;
  using PointType = typename Traits_::PointType;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  using IndexType = typename Traits_::IndexType;
  static SizeType constexpr Dim = Traits_::Dim;
  // TODO Temporary.
  using TraitsType = Traits_;

  explicit SpaceWrapper(SpaceType space) : space_(std::move(space)) {}

  inline SpaceType const& space() const { return space_; }
  inline constexpr SizeType sdim() const { return Traits_::SpaceSdim(space_); }
  inline IndexType size() const { return Traits_::SpaceNpts(space_); }

  inline auto PointAt(IndexType const idx) const {
    return Traits_::PointAt(space_, idx);
  }

  inline ScalarType const* PointCoordsAt(IndexType const idx) const {
    return Traits_::PointCoords(Traits_::PointAt(space_, idx));
  }

  inline ScalarType const& PointCoordAt(
      IndexType const point_idx, SizeType const coord_idx) const {
    return Traits_::PointCoords(Traits_::PointAt(space_, point_idx))[coord_idx];
  }

  inline Box<ScalarType, Dim> ComputeBoundingBox() const {
    Box<ScalarType, Dim> box(sdim());
    box.FillInverseMax();
    for (IndexType i = 0; i < size(); ++i) {
      box.Fit(PointCoordsAt(i));
    }
    return box;
  }

 private:
  SpaceType space_;
};

}  // namespace internal

}  // namespace pico_tree
