#pragma once

#include "py_array_map.hpp"

namespace pyco_tree {

template <typename Traits>
using L1 = pico_tree::L1<Traits>;

template <typename Traits>
using L2Squared = pico_tree::L2Squared<Traits>;

using PointsXf = PyArrayMap<float, pico_tree::kDynamicDim>;
using PointsXd = PyArrayMap<double, pico_tree::kDynamicDim>;
using Points2f = PyArrayMap<float, 2>;
using Points2d = PyArrayMap<double, 2>;
using Points3f = PyArrayMap<float, 3>;
using Points3d = PyArrayMap<double, 3>;

template <typename PointsX>
using TraitsX = MapTraits<
    typename PointsX::ScalarType,
    PointsX::Dim,
    typename PointsX::IndexType>;

using Neighborf = pico_tree::Neighbor<int, float>;
using Neighbord = pico_tree::Neighbor<int, double>;

}  // namespace pyco_tree
