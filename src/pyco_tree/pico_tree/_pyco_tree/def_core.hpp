#pragma once

#include "py_array_map.hpp"

namespace pyco_tree {

using PointsXf = PyArrayMap<float, pico_tree::kDynamicSize>;
using PointsXd = PyArrayMap<double, pico_tree::kDynamicSize>;
using Points2f = PyArrayMap<float, 2>;
using Points2d = PyArrayMap<double, 2>;
using Points3f = PyArrayMap<float, 3>;
using Points3d = PyArrayMap<double, 3>;

template <typename PointsX>
using TraitsX = PyArrayMapTraits<
    typename PointsX::ScalarType,
    PointsX::Dim,
    typename PointsX::IndexType>;

using Neighborf = pico_tree::Neighbor<int, float>;
using Neighbord = pico_tree::Neighbor<int, double>;

}  // namespace pyco_tree
