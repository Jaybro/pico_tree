#pragma once

#include "map.hpp"

namespace pyco_tree {

template <typename Traits>
using L1 = pico_tree::L1<Traits>;

template <typename Traits>
using L2Squared = pico_tree::L2Squared<Traits>;

using PointsXf = Map<int, float, pico_tree::kDynamicDim>;
using PointsXd = Map<int, double, pico_tree::kDynamicDim>;
using Points2f = Map<int, float, 2>;
using Points2d = Map<int, double, 2>;
using Points3f = Map<int, float, 3>;
using Points3d = Map<int, double, 3>;

template <typename PointsX>
using TraitsX = MapTraits<
    typename PointsX::IndexType,
    typename PointsX::ScalarType,
    PointsX::Dim>;

using Neighborf = pico_tree::Neighbor<int, float>;
using Neighbord = pico_tree::Neighbor<int, double>;

}  // namespace pyco_tree
