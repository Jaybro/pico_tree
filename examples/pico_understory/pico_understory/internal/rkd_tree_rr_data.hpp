#pragma once

#include <Eigen/Dense>

#include "eigen3.hpp"
#include "pico_tree/eigen3_traits.hpp"
#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/space_wrapper.hpp"

namespace pico_tree::internal {

//! \brief Sample from the uniform distribution on the Stiefel manifold (the set
//! of all orthonormal k-frames in R^n).
//! \details Sample = Z(Z^TZ)^(-1/2). Z has elements drawn from N(0,1).
//! The matrix may be improper.
template <typename Scalar_>
inline Eigen::Matrix<Scalar_, Eigen::Dynamic, Eigen::Dynamic>
RandomOrthogonalMatrix(pico_tree::Size dim) {
  // Working with floats makes all but JacobiSVD fail. Results for all are not
  // that accurate as well ((m * m.transpose()).diagonal().sum()).
  // * JacobiSVD<,Eigen::NoQRPreconditioner> never failed but slow.
  // * BDCSVD / bdcSvd may lose a rank or 2.
  // * SelfAdjointEigenSolver may lose 1 rank.
  // Solution: don't use float.
  // Reproduce: Matrix S = d.matrixU() *
  //            d.singularValues().cwiseInverse().cwiseSqrt().asDiagonal() *
  //            d.matrixU().transpose();
  //            // Eigen::DecompositionOptions::ComputeThinU
  using T = double;
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

  std::random_device rd;
  std::mt19937 e(rd());
  std::normal_distribution<T> gaussian(T(0), T(1));

  Matrix X = Matrix::Zero(dim, dim).unaryExpr(
      [&gaussian, &e](T dummy) { return gaussian(e); });

  Matrix XtX = X.transpose() * X;
  Eigen::SelfAdjointEigenSolver<Matrix> d(XtX);
  Matrix S = d.operatorInverseSqrt();
  // Returns the frames per row. Doesn't matter.
  if constexpr (std::is_same_v<Scalar_, T>) {
    return X * S;
  } else {
    return (X * S).cast<Scalar_>();
  }
}

template <typename Node_, Size Dim_>
class RKdTreeRrData {
  template <typename T, int Cols_ = PicoDimToEigenDim(Dim_)>
  using VectorType = VectorDX<T, Cols_>;

 public:
  using ScalarType = typename Node_::ScalarType;
  static Size constexpr Dim = Dim_;
  using NodeType = Node_;
  using RotationType =
      Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
  using SpaceType =
      Eigen::Matrix<ScalarType, PicoDimToEigenDim(Dim_), Eigen::Dynamic>;
  using SpaceWrapperType = SpaceWrapper<SpaceType>;

  template <typename SpaceWrapper_>
  static inline auto RandomRotation(SpaceWrapper_ space) {
    return RandomOrthogonalMatrix<ScalarType>(space.sdim());
  }

  template <typename SpaceWrapper_>
  static inline SpaceType RotateSpace(
      RotationType const& rotation, SpaceWrapper_ space) {
    SpaceType s(static_cast<int>(space.sdim()), static_cast<int>(space.size()));
    for (std::size_t i = 0; i < space.size(); ++i) {
      Eigen::Map<Eigen::Matrix<ScalarType, PicoDimToEigenDim(Dim_), 1> const> p(
          space[i], static_cast<int>(space.sdim()));
      s.col(i) = rotation * p;
    }
    return s;
  }

  template <typename PointWrapper_>
  VectorType<ScalarType> RotatePoint(PointWrapper_ w) const {
    Eigen::Map<VectorType<ScalarType> const> p(w.begin(), space.rows());
    return rotation * p;
  }

  RotationType rotation;
  SpaceType space;
  KdTreeData<Node_, Dim_> tree;
};

}  // namespace pico_tree::internal
