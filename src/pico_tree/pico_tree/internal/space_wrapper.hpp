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
  using PointTraitsType = typename Traits_::template PointTraitsFor<PointType>;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  using IndexType = typename Traits_::IndexType;
  static SizeType constexpr Dim = Traits_::Dim;

  explicit SpaceWrapper(SpaceType space) : space_(std::move(space)) {}

  inline ScalarType const* operator[](IndexType const idx) const {
    return PointTraitsType::Coords(Traits_::PointAt(space_, idx));
  }

  inline Box<ScalarType, Dim> ComputeBoundingBox() const {
    Box<ScalarType, Dim> box(sdim());
    box.FillInverseMax();
    for (IndexType i = 0; i < size(); ++i) {
      box.Fit(operator[](i));
    }
    return box;
  }

  constexpr SizeType sdim() const {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      return Traits_::Sdim(space_);
    }
  }

  inline IndexType size() const { return Traits_::Npts(space_); }

  inline SpaceType const& space() const { return space_; }

 private:
  SpaceType space_;
};

}  // namespace internal

}  // namespace pico_tree
