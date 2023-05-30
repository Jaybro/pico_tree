#pragma once

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

// TODO: Point_ is temporary
template <typename Traits_, typename Point_>
class PointWrapper {
 public:
  using PointType = Point_;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  static SizeType constexpr Dim = Traits_::Dim;

  explicit PointWrapper(PointType const& point) : point_(point) {}

  inline ScalarType const& operator[](std::size_t index) const {
    return data()[index];
  }

  inline ScalarType const* data() const { return Traits_::PointCoords(point_); }
  constexpr SizeType size() const {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      return Traits_::PointSdim(point_);
    }
  }
  inline PointType const& point() const { return point_; }

 private:
  PointType const& point_;
};

}  // namespace internal

}  // namespace pico_tree
