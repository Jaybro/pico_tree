#include <gtest/gtest.h>

#include <pico_tree/eigen3_traits.hpp>

#include "common.hpp"

template <typename ColMatrix, typename RowMatrix>
void CheckEigenAdaptorInterface() {
  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);

  CheckSpaceAdaptor<static_cast<pico_tree::Size>(ColMatrix::RowsAtCompileTime)>(
      std::cref(col_matrix),
      col_matrix.rows(),
      col_matrix.cols(),
      col_matrix.cols() - 1,
      col_matrix.col(col_matrix.cols() - 1).data());
  CheckSpaceAdaptor<static_cast<pico_tree::Size>(RowMatrix::ColsAtCompileTime)>(
      std::cref(row_matrix),
      row_matrix.cols(),
      row_matrix.rows(),
      row_matrix.rows() - 1,
      row_matrix.row(row_matrix.rows() - 1).data());
}

TEST(Eigen3TraitsTest, Interface) {
  // Spatial dimension known.
  CheckEigenAdaptorInterface<
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>>();
  // Spatial dimension dynamic.
  CheckEigenAdaptorInterface<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>();
}
