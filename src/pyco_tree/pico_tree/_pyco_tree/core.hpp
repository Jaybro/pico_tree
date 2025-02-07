#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/metric.hpp>

namespace py = pybind11;

namespace pyco_tree {

template <typename T_>
struct string_traits;

template <>
struct string_traits<float> {
  static std::string type_string() { return "float32"; }
};

template <>
struct string_traits<double> {
  static std::string type_string() { return "float64"; }
};

template <>
struct string_traits<pico_tree::metric_l1> {
  static std::string type_string() { return "L1"; }
};

template <>
struct string_traits<pico_tree::metric_l2_squared> {
  static std::string type_string() { return "L2Squared"; }
};

template <>
struct string_traits<pico_tree::metric_linf> {
  static std::string type_string() { return "LInf"; }
};

template <typename T_>
bool is_row_major(py::array_t<T_, 0> const& array) {
  return (array.flags() & py::array::c_style) > 0;
}

template <typename T_>
bool is_col_major(py::array_t<T_, 0> const& array) {
  return (array.flags() & py::array::f_style) > 0;
}

template <typename T_>
bool is_contiguous(py::array_t<T_, 0> const& array) {
  return is_row_major(array) || is_col_major(array);
}

template <typename T_>
struct array_layout {
  array_layout(py::array_t<T_, 0> const& array)
      : info(array.request()),
        row_major(is_row_major(array)),
        index_inner(row_major ? 1 : 0),
        index_outer(row_major ? 0 : 1) {}

  py::buffer_info info;
  bool row_major;
  py::ssize_t index_inner;
  py::ssize_t index_outer;
};

template <pico_tree::size_t Dim_>
inline bool is_dim_compatible(pico_tree::size_t d) {
  return Dim_ == d;
}

template <>
inline bool is_dim_compatible<pico_tree::dynamic_size>(pico_tree::size_t) {
  return true;
}

}  // namespace pyco_tree
