#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/core.hpp>

#include "core.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename Index, typename Scalar, int Dim_>
class PycoAdaptor {
 public:
  using IndexType = Index;
  using ScalarType = Scalar;
  static constexpr int Dim = Dim_;

  class Point {
   public:
    inline explicit Point(Scalar const* const data) : data_(data) {}

    inline Scalar const& operator()(int const i) const { return data_[i]; }

   private:
    Scalar const* const data_;
  };

  PycoAdaptor(py::array_t<Scalar, 0> const pts) {
    ArrayLayout<Scalar> layout(pts);

    if (layout.info.ndim != 2) {
      throw std::runtime_error("Array: ndim not 2.");
    }

    // We always want the memory layout to look like x,y,z,x,y,z,...,x,y,z.
    // This means that the shape of the inner dimension should equal the spatial
    // dimension of the KdTree.
    if (!IsDimCompatible<Dim>(
            static_cast<int>(layout.info.shape[layout.index_inner]))) {
      throw std::runtime_error(
          "Array: Incompatible KdTree sdim and Array inner stride.");
    }

    ThrowIfNotContiguous(layout);

    data_ = static_cast<Scalar*>(layout.info.ptr);
    sdim_ = static_cast<int>(layout.info.shape[layout.index_inner]);
    npts_ = static_cast<Index>(layout.info.shape[layout.index_outer]);
    row_major_ = layout.row_major;
  }

  inline Point operator()(Index const idx) const {
    return Point(data_ + idx * sdim_);
  }

  inline Scalar const* const data() const { return data_; }

  inline int sdim() const { return sdim_; }

  inline Index npts() const { return npts_; }

  inline bool row_major() const { return row_major_; }

 private:
  Scalar const* data_;
  int sdim_;
  Index npts_;
  bool row_major_;
};

}  // namespace pyco_tree
