#pragma once

#include "pico_tree/core.hpp"

namespace pico_tree {

namespace internal {

template <typename Traits_>
class PointWrapper {
 public:
  using PointType = typename Traits_::PointType;
  using ScalarType = typename Traits_::ScalarType;
  using SizeType = Size;
  static SizeType constexpr Dim = Traits_::Dim;

  inline explicit PointWrapper(PointType const& point) : point_(point) {}

  inline ScalarType const& operator[](std::size_t index) const {
    return data()[index];
  }

  inline ScalarType const* data() const { return Traits_::Coords(point_); }

  constexpr SizeType size() const {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      return Traits_::Sdim(point_);
    }
  }

  inline PointType const& point() const { return point_; }

 private:
  PointType const& point_;
};

}  // namespace internal

}  // namespace pico_tree
