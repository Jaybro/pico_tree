#pragma once

#include "py_array_map.hpp"

namespace pyco_tree {

using space_xf = py_array_map<float, pico_tree::dynamic_size>;
using space_xd = py_array_map<double, pico_tree::dynamic_size>;
using space_2f = py_array_map<float, 2>;
using space_2d = py_array_map<double, 2>;
using space_3f = py_array_map<float, 3>;
using space_3d = py_array_map<double, 3>;

template <typename Points_>
using space_x = py_array_map<typename Points_::scalar_type, Points_::dim>;

using neighbor_f = pico_tree::neighbor<int, float>;
using neighbor_d = pico_tree::neighbor<int, double>;

}  // namespace pyco_tree
