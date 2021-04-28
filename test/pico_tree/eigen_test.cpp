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
  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);
  using ValColTraits = pico_tree::EigenTraits<ColMatrix>;
  using ValRowTraits = pico_tree::EigenTraits<RowMatrix>;
  using RefColTraits =
      pico_tree::EigenTraits<std::reference_wrapper<ColMatrix>, std::size_t>;
  using RefRowTraits = pico_tree::
      EigenTraits<std::reference_wrapper<RowMatrix const>, std::size_t>;

  CheckTraits<ValColTraits, ColMatrix::RowsAtCompileTime, int>(
      col_matrix,
      col_matrix.rows(),
      col_matrix.cols(),
      static_cast<Eigen::Index>(0),
      col_matrix.col(0).data());
  CheckTraits<ValRowTraits, RowMatrix::ColsAtCompileTime, int>(
      row_matrix,
      row_matrix.cols(),
      row_matrix.rows(),
      static_cast<Eigen::Index>(0),
      row_matrix.row(0).data());

  CheckTraits<RefColTraits, ColMatrix::RowsAtCompileTime, std::size_t>(
      std::ref(col_matrix),
      col_matrix.rows(),
      col_matrix.cols(),
      col_matrix.cols() - 1,
      col_matrix.col(col_matrix.cols() - 1).data());
  CheckTraits<RefRowTraits, RowMatrix::ColsAtCompileTime, std::size_t>(
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

  pico_tree::KdTree<pico_tree::EigenTraits<Eigen::Matrix4Xd>> tree(
      std::move(matrix), 10);

  TestKnn(tree, 2);
}
