#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/std_traits.hpp>

#include "core.hpp"
#include "map_straits.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename Scalar_, int Dim_>
class Map {
 public:
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;
  // Fixed to be an int.
  using IndexType = int;

  inline Map(
      ScalarType* data, std::size_t npts, std::size_t sdim, bool row_major)
      : space_(data, npts, sdim), row_major_(row_major) {}

  inline pico_tree::PointMap<ScalarType const, Dim> operator()(
      std::size_t i) const {
    return space_(i);
  }

  inline ScalarType const* data() const { return space_.data(); }
  inline ScalarType* data() { return space_.data(); }
  inline std::size_t npts() const { return space_.npts(); }
  inline std::size_t sdim() const { return space_.sdim(); }
  inline bool row_major() const { return row_major_; }

 private:
  pico_tree::SpaceMap<ScalarType, Dim> space_;
  bool row_major_;
};

template <typename Scalar_, int Dim_>
Map<Scalar_, Dim_> MakeMap(py::array_t<Scalar_, 0> const pts) {
  ArrayLayout<Scalar_> layout(pts);

  if (layout.info.ndim != 2) {
    throw std::runtime_error("Array: ndim not 2.");
  }

  // We always want the memory layout to look like x,y,z,x,y,z,...,x,y,z.
  // This means that the shape of the inner dimension should equal the spatial
  // dimension of the KdTree.
  if (!IsDimCompatible<Dim_>(
          static_cast<int>(layout.info.shape[layout.index_inner]))) {
    throw std::runtime_error(
        "Array: Incompatible KdTree sdim and Array inner stride.");
  }

  ThrowIfNotContiguous(layout);

  return Map<Scalar_, Dim_>(
      static_cast<Scalar_*>(layout.info.ptr),
      static_cast<std::size_t>(layout.info.shape[layout.index_outer]),
      static_cast<std::size_t>(layout.info.shape[layout.index_inner]),
      layout.row_major);
}

template <typename Scalar_, int Dim_, typename Index_>
struct MapTraits {
  using SpaceType = Map<Scalar_, Dim_>;
  using PointType = pico_tree::PointMap<Scalar_ const, Dim_>;
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;
  using IndexType = Index_;

  inline static int SpaceSdim(SpaceType const& space) {
    return static_cast<IndexType>(space.sdim());
  }

  inline static IndexType SpaceNpts(SpaceType const& space) {
    return static_cast<IndexType>(space.npts());
  }

  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space(idx);
  }

  template <typename OtherPoint>
  inline static int PointSdim(OtherPoint const& point) {
    return pico_tree::StdPointTraits<OtherPoint>::Sdim(point);
  }

  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return pico_tree::StdPointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pyco_tree
