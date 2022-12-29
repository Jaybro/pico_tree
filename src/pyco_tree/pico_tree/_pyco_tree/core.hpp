#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/metric.hpp>

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

template <typename Traits>
struct StringTraits<pico_tree::L1<Traits>> {
  static std::string String() { return "L1"; }
};

template <typename Traits>
struct StringTraits<pico_tree::L2Squared<Traits>> {
  static std::string String() { return "L2Squared"; }
};

template <typename T>
bool IsRowMajor(py::array_t<T, 0> const& array) {
  return (array.flags() & py::array::c_style) > 0;
}

template <typename T>
bool IsColMajor(py::array_t<T, 0> const& array) {
  return (array.flags() & py::array::f_style) > 0;
}

template <typename T>
bool IsContiguous(py::array_t<T, 0> const& array) {
  return IsRowMajor(array) || IsColMajor(array);
}

template <typename T>
struct ArrayLayout {
  ArrayLayout(py::array_t<T, 0> const& array)
      : info(array.request()),
        row_major(IsRowMajor(array)),
        index_inner(row_major ? 1 : 0),
        index_outer(row_major ? 0 : 1) {}

  py::buffer_info info;
  bool row_major;
  py::ssize_t index_inner;
  py::ssize_t index_outer;
};

template <pico_tree::Size Dim>
inline bool IsDimCompatible(pico_tree::Size dim) {
  return Dim == dim;
}

template <>
inline bool IsDimCompatible<pico_tree::kDynamicDim>(pico_tree::Size) {
  return true;
}

}  // namespace pyco_tree
