#include <gtest/gtest.h>

#include <pico_tree/eigen3_traits.hpp>

#include "common.hpp"

template <typename ColMatrix_, typename RowMatrix_>
void check_eigen_adaptor_interface() {
  ColMatrix_ col_matrix = ColMatrix_::Random(4, 8);
  RowMatrix_ row_matrix = RowMatrix_::Random(4, 8);

  check_space_adaptor<static_cast<pico_tree::size_t>(
      ColMatrix_::RowsAtCompileTime)>(
      std::cref(col_matrix),
      static_cast<pico_tree::size_t>(col_matrix.rows()),
      static_cast<pico_tree::size_t>(col_matrix.cols()),
      static_cast<pico_tree::size_t>(col_matrix.cols() - 1),
      col_matrix.col(col_matrix.cols() - 1).data());
  check_space_adaptor<static_cast<pico_tree::size_t>(
      RowMatrix_::ColsAtCompileTime)>(
      std::cref(row_matrix),
      static_cast<pico_tree::size_t>(row_matrix.cols()),
      static_cast<pico_tree::size_t>(row_matrix.rows()),
      static_cast<pico_tree::size_t>(row_matrix.rows() - 1),
      row_matrix.row(row_matrix.rows() - 1).data());
}

TEST(Eigen3TraitsTest, Interface) {
  // Spatial dimension known.
  check_eigen_adaptor_interface<
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>>();
  // Spatial dimension dynamic.
  check_eigen_adaptor_interface<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>();
}
