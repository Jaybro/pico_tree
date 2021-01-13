#pragma once

#include <pybind11/pybind11.h>

#include <pico_tree/kd_tree.hpp>
#include <vector>

#include "pyco_adaptor.hpp"

namespace pyco_tree {

template <typename Points>
using MetricL1 = pico_tree::MetricL1<typename Points::ScalarType, Points::Dim>;

template <typename Points>
using MetricL2 = pico_tree::MetricL2<typename Points::ScalarType, Points::Dim>;

using PointsXf = PycoAdaptor<int, float, pico_tree::kDynamicDim>;
using PointsXd = PycoAdaptor<int, double, pico_tree::kDynamicDim>;
using Points2f = PycoAdaptor<int, float, 2>;
using Points2d = PycoAdaptor<int, double, 2>;
using Points3f = PycoAdaptor<int, float, 3>;
using Points3d = PycoAdaptor<int, double, 3>;

using Neighborf = pico_tree::Neighbor<int, float>;
using Neighbord = pico_tree::Neighbor<int, double>;

using Neighborhoodf = std::vector<Neighborf>;
using Neighborhoodd = std::vector<Neighbord>;

using Neighborhoodsf = std::vector<Neighborhoodf>;
using Neighborhoodsd = std::vector<Neighborhoodd>;

void DefCore(pybind11::module* m);

}  // namespace pyco_tree
