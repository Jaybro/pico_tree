#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/std_traits.hpp>

#include "core.hpp"
#include "map_straits.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief The Map class adds the row_major property to the pico_tree::SpaceMap
//! class.
template <typename Scalar_, int Dim_>
class Map : public pico_tree::SpaceMap<Scalar_, Dim_> {
 public:
  using typename pico_tree::SpaceMap<Scalar_, Dim_>::ScalarType;
  using pico_tree::SpaceMap<Scalar_, Dim_>::Dim;
  // Fixed to be an int.
  using IndexType = int;

  inline Map(
      ScalarType* data, std::size_t npts, std::size_t sdim, bool row_major)
      : pico_tree::SpaceMap<Scalar_, Dim_>(data, npts, sdim),
        row_major_(row_major) {}

  inline bool row_major() const { return row_major_; }

 private:
  bool row_major_;
};

template <typename Scalar_, int Dim_>
Map<Scalar_, Dim_> MakeMap(py::array_t<Scalar_, 0> const pts) {
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
  if (!IsDimCompatible<Dim_>(
          static_cast<int>(layout.info.shape[layout.index_inner]))) {
    throw std::runtime_error(
        "Array: Incompatible KdTree sdim and Array inner stride.");
  }

  return Map<Scalar_, Dim_>(
      static_cast<Scalar_*>(layout.info.ptr),
      static_cast<std::size_t>(layout.info.shape[layout.index_outer]),
      static_cast<std::size_t>(layout.info.shape[layout.index_inner]),
      layout.row_major);
}

template <typename Scalar_, int Dim_, typename Index_>
struct MapTraits : public pico_tree::MapTraits<Scalar_, Dim_, Index_> {
  using SpaceType = Map<Scalar_, Dim_>;
};

}  // namespace pyco_tree
