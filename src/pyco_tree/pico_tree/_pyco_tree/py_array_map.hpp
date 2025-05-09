#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/map_traits.hpp>
#include <stdexcept>

#include "core.hpp"

namespace py = pybind11;

namespace pyco_tree {

//! \brief The py_array_map class adds the row_major property to the
//! pico_tree::space_map class.
template <typename Scalar_, pico_tree::size_t Dim_>
class py_array_map
    : public pico_tree::space_map<pico_tree::point_map<Scalar_, Dim_>> {
 public:
  using point_type = pico_tree::point_map<Scalar_, Dim_>;
  using typename pico_tree::space_map<point_type>::scalar_type;
  using typename pico_tree::space_map<point_type>::size_type;
  using pico_tree::space_map<point_type>::dim;

  inline py_array_map(
      Scalar_* data, size_type npts, size_type sdim, bool row_major)
      : pico_tree::space_map<point_type>(data, npts, sdim),
        row_major_(row_major) {}

  inline bool row_major() const { return row_major_; }

 private:
  bool row_major_;
};

template <pico_tree::size_t Dim_, typename Scalar_>
py_array_map<Scalar_, Dim_> make_map(py::array const space) {
  array_layout layout(space);

  if (layout.info.ndim != 2) {
    throw std::invalid_argument("array ndim not 2");
  }

  if (!is_contiguous(space)) {
    throw std::invalid_argument("array not contiguous");
  }

  // We always want the memory layout to look like x,y,z,x,y,z,...,x,y,z.
  // This means that the shape of the inner dimension should equal the spatial
  // dimension of the kd_tree.
  if (!is_dim_compatible<Dim_>(
          static_cast<pico_tree::size_t>(layout.inner_stride()))) {
    throw std::invalid_argument(
        "incompatible kd_tree sdim and array inner stride");
  }

  return py_array_map<Scalar_, Dim_>(
      static_cast<Scalar_*>(layout.info.ptr),
      static_cast<pico_tree::size_t>(layout.outer_stride()),
      static_cast<pico_tree::size_t>(layout.inner_stride()),
      layout.row_major);
}

}  // namespace pyco_tree

namespace pico_tree {

template <typename Scalar_, size_t Dim_>
struct space_traits<pyco_tree::py_array_map<Scalar_, Dim_>>
    : public space_traits<space_map<point_map<Scalar_, Dim_>>> {
  using space_type = pyco_tree::py_array_map<Scalar_, Dim_>;
};

}  // namespace pico_tree
