#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/map_traits.hpp>

#include "core.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief The Map class adds the row_major property to the pico_tree::SpaceMap
//! class.
template <typename Scalar_, pico_tree::Size Dim_>
class PyArrayMap
    : public pico_tree::SpaceMap<pico_tree::PointMap<Scalar_, Dim_>> {
 public:
  using PointType = pico_tree::PointMap<Scalar_, Dim_>;
  using typename pico_tree::SpaceMap<PointType>::ScalarType;
  using typename pico_tree::SpaceMap<PointType>::SizeType;
  using pico_tree::SpaceMap<PointType>::Dim;
  // Fixed to be an int.
  using IndexType = int;

  inline PyArrayMap(Scalar_* data, SizeType npts, SizeType sdim, bool row_major)
      : pico_tree::SpaceMap<PointType>(data, npts, sdim),
        row_major_(row_major) {}

  inline bool row_major() const { return row_major_; }

 private:
  bool row_major_;
};

template <pico_tree::Size Dim_, typename Scalar_>
PyArrayMap<Scalar_, Dim_> MakeMap(py::array_t<Scalar_, 0> const pts) {
  ArrayLayout<Scalar_> layout(pts);

  if (layout.info.ndim != 2) {
    throw std::runtime_error("Array: ndim not 2.");
  }

  if (!IsContiguous(pts)) {
    throw std::runtime_error("Array: Array not contiguous.");
  }

  // We always want the memory layout to look like x,y,z,x,y,z,...,x,y,z.
  // This means that the shape of the inner dimension should equal the spatial
  // dimension of the KdTree.
  if (!IsDimCompatible<Dim_>(static_cast<pico_tree::Size>(
          layout.info.shape[layout.index_inner]))) {
    throw std::runtime_error(
        "Array: Incompatible KdTree sdim and Array inner stride.");
  }

  return PyArrayMap<Scalar_, Dim_>(
      static_cast<Scalar_*>(layout.info.ptr),
      static_cast<pico_tree::Size>(layout.info.shape[layout.index_outer]),
      static_cast<pico_tree::Size>(layout.info.shape[layout.index_inner]),
      layout.row_major);
}

}  // namespace pyco_tree

namespace pico_tree {

template <typename Scalar_, pico_tree::Size Dim_>
struct SpaceTraits<pyco_tree::PyArrayMap<Scalar_, Dim_>>
    : public SpaceTraits<SpaceMap<pico_tree::PointMap<Scalar_, Dim_>>> {
  using SpaceType = pyco_tree::PyArrayMap<Scalar_, Dim_>;
};

}  // namespace pico_tree
