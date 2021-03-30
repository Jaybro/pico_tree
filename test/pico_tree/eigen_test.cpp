#include <gtest/gtest.h>

#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>

#include "common.hpp"

namespace {

using Index = int;

template <typename SpaceX>
using Traits = pico_tree::EigenTraits<SpaceX>;

}  // namespace

TEST(EigenTest, EigenL1) {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;

  PointX p0{10.0f, 1.0f};
  PointX p1{2.0f, 4.0f};

  pico_tree::EigenL1<Scalar> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(EigenTest, EigenL2) {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;

  PointX p0{10.0f, 1.0f};
  PointX p1{7.0f, 5.0f};

  pico_tree::EigenL2<Scalar> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(EigenTest, EigenL2Squared) {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;

  PointX p0{10.0f, 1.0f};
  PointX p1{2.0f, 4.0f};

  pico_tree::EigenL2Squared<Scalar> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}

template <typename ColMatrix, typename RowMatrix>
void CheckEigenAdaptorInterface() {
  static_assert(
      pico_tree::EigenTraits<ColMatrix>::Dim == ColMatrix::RowsAtCompileTime,
      "ADAPTOR_DIM_NOT_EQUAL_TO_MATRIX_ROWSATCOMPILETIME");

  static_assert(
      pico_tree::EigenTraits<RowMatrix>::Dim == ColMatrix::ColsAtCompileTime,
      "ADAPTOR_DIM_NOT_EQUAL_TO_MATRIX_COLSATCOMPILETIME");

  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);

  EXPECT_EQ(
      col_matrix.rows(),
      pico_tree::EigenTraits<ColMatrix>::SpaceSdim(col_matrix));
  EXPECT_EQ(
      col_matrix.cols(),
      pico_tree::EigenTraits<ColMatrix>::SpaceNpts(col_matrix));
  EXPECT_EQ(
      row_matrix.cols(),
      pico_tree::EigenTraits<RowMatrix>::SpaceSdim(row_matrix));
  EXPECT_EQ(
      row_matrix.rows(),
      pico_tree::EigenTraits<RowMatrix>::SpaceNpts(row_matrix));

  EXPECT_TRUE(pico_tree::EigenTraits<ColMatrix>::PointAt(col_matrix, 0)
                  .isApprox(col_matrix.col(0)));
  EXPECT_TRUE(pico_tree::EigenTraits<ColMatrix>::PointAt(
                  col_matrix, col_matrix.cols() - 1)
                  .isApprox(col_matrix.col(col_matrix.cols() - 1)));
  EXPECT_TRUE(pico_tree::EigenTraits<RowMatrix>::PointAt(row_matrix, 0)
                  .isApprox(row_matrix.row(0)));
  EXPECT_TRUE(pico_tree::EigenTraits<RowMatrix>::PointAt(
                  row_matrix, row_matrix.rows() - 1)
                  .isApprox(row_matrix.row(row_matrix.rows() - 1)));
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

  pico_tree::KdTree<pico_tree::EigenTraits<Eigen::Matrix4Xd>> tree(
      std::move(matrix), 10);

  TestKnn(tree, 2);
}
