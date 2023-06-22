#include <gtest/gtest.h>

#include <pico_tree/eigen3_traits.hpp>
#include <pico_tree/kd_tree.hpp>

#include "common.hpp"

namespace {

using Index = int;

template <typename SpaceX>
using Traits = pico_tree::SpaceTraits<SpaceX>;

}  // namespace

template <typename ColMatrix, typename RowMatrix>
void CheckEigenAdaptorInterface() {
  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);
  using ValColTraits = pico_tree::SpaceTraits<ColMatrix>;
  using ValRowTraits = pico_tree::SpaceTraits<RowMatrix>;
  using RefColTraits =
      pico_tree::SpaceTraits<std::reference_wrapper<ColMatrix>>;
  using RefRowTraits =
      pico_tree::SpaceTraits<std::reference_wrapper<RowMatrix const>>;

  CheckTraits<
      ValColTraits,
      static_cast<pico_tree::Size>(ColMatrix::RowsAtCompileTime)>(
      col_matrix,
      col_matrix.rows(),
      col_matrix.cols(),
      static_cast<Eigen::Index>(0),
      col_matrix.col(0).data());
  CheckTraits<
      ValRowTraits,
      static_cast<pico_tree::Size>(RowMatrix::ColsAtCompileTime)>(
      row_matrix,
      row_matrix.cols(),
      row_matrix.rows(),
      static_cast<Eigen::Index>(0),
      row_matrix.row(0).data());

  CheckTraits<
      RefColTraits,
      static_cast<pico_tree::Size>(ColMatrix::RowsAtCompileTime)>(
      std::ref(col_matrix),
      col_matrix.rows(),
      col_matrix.cols(),
      col_matrix.cols() - 1,
      col_matrix.col(col_matrix.cols() - 1).data());
  CheckTraits<
      RefRowTraits,
      static_cast<pico_tree::Size>(RowMatrix::ColsAtCompileTime)>(
      std::cref(row_matrix),
      row_matrix.cols(),
      row_matrix.rows(),
      col_matrix.rows() - 1,
      row_matrix.row(row_matrix.rows() - 1).data());
}

TEST(EigenTest, Interface) {
  // Spatial dimension known.
  CheckEigenAdaptorInterface<
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>>();
  // Spatial dimension dynamic.
  CheckEigenAdaptorInterface<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>();
}

TEST(EigenTest, TreeCompatibility) {
  Eigen::Matrix4Xd matrix = Eigen::Matrix4Xd::Random(4, 8);

  pico_tree::KdTree<Eigen::Matrix4Xd> tree(std::move(matrix), 10);

  TestKnn(tree, 2);
}
