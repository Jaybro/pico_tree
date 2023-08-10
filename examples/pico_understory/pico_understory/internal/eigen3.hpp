#pragma once

#include <Eigen/Core>

#include "pico_tree/core.hpp"

namespace pico_tree::internal {

constexpr int PicoDimToEigenDim(pico_tree::Size dim) {
  return dim == pico_tree::kDynamicSize ? Eigen::Dynamic
                                        : static_cast<int>(dim);
}

template <typename Scalar_, int Dim_>
using VectorDX = Eigen::Matrix<Scalar_, Dim_, 1>;

}  // namespace pico_tree::internal
