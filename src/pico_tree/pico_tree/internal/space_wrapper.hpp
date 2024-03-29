#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/box.hpp"
#include "pico_tree/space_traits.hpp"

namespace pico_tree::internal {

//! \brief The SpaceWrapper class wraps makes working with any space type
//! through its respective SpaceTraits a bit easier and it allows for the
//! addition of extra convenience methods.
//! \details The internals of PicoTree never use the specializations of the
//! SpaceTraits class directly, but interface with any space type through this
//! wrapper interface
template <typename Space_>
class SpaceWrapper {
  using SpaceTraitsType = SpaceTraits<Space_>;
  using SpaceType = Space_;
  using PointType = typename SpaceTraitsType::PointType;
  using PointTraitsType = PointTraits<PointType>;
  using SizeType = Size;

 public:
  using ScalarType = typename SpaceTraitsType::ScalarType;
  static SizeType constexpr Dim = SpaceTraitsType::Dim;

  explicit SpaceWrapper(SpaceType const& space) : space_(space) {}

  template <typename Index_>
  inline ScalarType const* operator[](Index_ const index) const {
    return PointTraitsType::data(SpaceTraitsType::PointAt(space_, index));
  }

  inline Box<ScalarType, Dim> ComputeBoundingBox() const {
    Box<ScalarType, Dim> box(sdim());
    box.FillInverseMax();
    for (SizeType i = 0; i < size(); ++i) {
      box.Fit(operator[](i));
    }
    return box;
  }

  inline SizeType size() const { return SpaceTraitsType::size(space_); }

  constexpr SizeType sdim() const {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      return SpaceTraitsType::sdim(space_);
    }
  }

 private:
  SpaceType const& space_;
};

}  // namespace pico_tree::internal
