#pragma once

#include "map.hpp"

namespace pyco_tree {

template <typename Traits>
using L1 = pico_tree::L1<Traits>;

template <typename Traits>
using L2Squared = pico_tree::L2Squared<Traits>;

using PointsXf = Map<float, pico_tree::kDynamicDim>;
using PointsXd = Map<double, pico_tree::kDynamicDim>;
using Points2f = Map<float, 2>;
using Points2d = Map<double, 2>;
using Points3f = Map<float, 3>;
using Points3d = Map<double, 3>;

template <typename PointsX>
using TraitsX = MapTraits<
    typename PointsX::ScalarType,
    PointsX::Dim,
    typename PointsX::IndexType>;

using Neighborf = pico_tree::Neighbor<int, float>;
using Neighbord = pico_tree::Neighbor<int, double>;

}  // namespace pyco_tree
