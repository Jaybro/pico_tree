#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/core.hpp>

namespace py = pybind11;

namespace pyco_tree {

template <typename T>
struct StringTraits;

template <>
struct StringTraits<float> {
  static std::string String() { return "float32"; }
};

template <>
struct StringTraits<double> {
  static std::string String() { return "float64"; }
};

template <typename Scalar, int Dim>
struct StringTraits<pico_tree::MetricL1<Scalar, Dim>> {
  static std::string String() { return "L1"; }
};

template <typename Scalar, int Dim>
struct StringTraits<pico_tree::MetricL2<Scalar, Dim>> {
  static std::string String() { return "L2"; }
};

template <typename T>
bool IsRowMajor(py::array_t<T, 0> const array) {
  return (array.flags() & py::array::c_style) > 0;
}

template <typename T>
struct ArrayLayout {
  ArrayLayout(py::array_t<T, 0> const array)
      : info(array.request()),
        row_major(IsRowMajor(array)),
        index_inner(row_major ? 1 : 0),
        index_outer(row_major ? 0 : 1) {}

  py::buffer_info info;
  bool row_major;
  py::ssize_t index_inner;
  py::ssize_t index_outer;
};

template <typename T>
void ThrowIfNotContiguous(ArrayLayout<T> const& layout) {
  if (layout.info.strides[layout.index_inner] != layout.info.itemsize) {
    throw std::runtime_error("Array: Inner stride not contiguous.");
  }

  if (layout.info.ndim == 2 &&
      layout.info.strides[layout.index_outer] !=
          layout.info.itemsize * layout.info.shape[layout.index_inner]) {
    throw std::runtime_error("Array: Outer stride not contiguous.");
  }

  // For ndim 3+ it would be the same for both layouts again. However, we
  // should not have to deal with this.
}

}  // namespace pyco_tree
