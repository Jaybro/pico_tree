#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/point_traits.hpp"

namespace pico_tree::internal {

//! \brief The PointWrapper class wraps makes working with any point type
//! through its respective PointTraits a bit easier and it allows for the
//! addition of extra convenience methods.
//! \details The internals of PicoTree never use the specializations of the
//! PointTraits class directly, but interface with any point type through this
//! wrapper interface.
template <typename Point_>
class PointWrapper {
  using PointTraitsType = PointTraits<Point_>;
  using PointType = Point_;
  using ScalarType = typename PointTraitsType::ScalarType;
  using SizeType = Size;
  static SizeType constexpr Dim = PointTraitsType::Dim;

  inline ScalarType const* data() const {
    return PointTraitsType::data(point_);
  }

  constexpr SizeType size() const {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      return PointTraitsType::size(point_);
    }
  }

 public:
  inline explicit PointWrapper(PointType const& point) : point_(point) {}

  inline ScalarType const& operator[](std::size_t index) const {
    return data()[index];
  }

  inline auto begin() const { return data(); }

  inline auto end() const { return data() + size(); }

 private:
  PointType const& point_;
};

}  // namespace pico_tree::internal
