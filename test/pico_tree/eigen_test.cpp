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

template <int Dim, typename Matrix>
void EigenCheckTypes(
    Matrix const& matrix, Eigen::Index sdim, Eigen::Index npts) {
  static_assert(
      std::is_same<typename pico_tree::EigenTraits<Matrix>::SpaceType, Matrix>::
          value,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(
      pico_tree::EigenTraits<Matrix>::Dim == Dim,
      "TRAITS_DIM_NOT_EQUAL_TO_EXPECTED_DIM");

  static_assert(
      std::is_same<typename pico_tree::EigenTraits<Matrix>::IndexType, int>::
          value,
      "TRAITS_INDEX_TYPE_NOT_INT");

  static_assert(
      std::is_same<
          typename pico_tree::EigenTraits<Matrix, std::size_t>::IndexType,
          std::size_t>::value,
      "TRAITS_INDEX_TYPE_NOT_SIZE_T");

  EXPECT_EQ(
      static_cast<int>(sdim),
      pico_tree::EigenTraits<Matrix>::SpaceSdim(matrix));
  EXPECT_EQ(
      static_cast<int>(npts),
      pico_tree::EigenTraits<Matrix>::SpaceNpts(matrix));
}

template <typename ColMatrix, typename RowMatrix>
void CheckEigenAdaptorInterface() {
  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);

  EigenCheckTypes<ColMatrix::RowsAtCompileTime>(
      col_matrix, col_matrix.rows(), col_matrix.cols());
  EigenCheckTypes<RowMatrix::ColsAtCompileTime>(
      row_matrix, row_matrix.cols(), row_matrix.rows());

  EigenCheckTypes<ColMatrix::RowsAtCompileTime>(
      std::ref(col_matrix), col_matrix.rows(), col_matrix.cols());
  EigenCheckTypes<RowMatrix::ColsAtCompileTime>(
      std::cref(row_matrix), row_matrix.cols(), row_matrix.rows());

  EXPECT_TRUE(pico_tree::EigenTraits<ColMatrix>::PointAt(col_matrix, 0)
                  .isApprox(col_matrix.col(0)));
  EXPECT_TRUE(pico_tree::EigenTraits<ColMatrix>::PointAt(
                  col_matrix, static_cast<int>(col_matrix.cols()) - 1)
                  .isApprox(col_matrix.col(col_matrix.cols() - 1)));
  EXPECT_TRUE(pico_tree::EigenTraits<RowMatrix>::PointAt(row_matrix, 0)
                  .isApprox(row_matrix.row(0)));
  EXPECT_TRUE(pico_tree::EigenTraits<RowMatrix>::PointAt(
                  row_matrix, static_cast<int>(row_matrix.rows()) - 1)
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
