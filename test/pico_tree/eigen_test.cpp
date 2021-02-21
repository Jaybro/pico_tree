#include <gtest/gtest.h>

#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>

#include "common.hpp"

using Index = int;

template <typename Point>
using EigenMap = Eigen::Map<
    Eigen::Matrix<
        typename Point::Scalar,
        Point::RowsAtCompileTime,
        Eigen::Dynamic>,
    Eigen::AlignedMax>;

template <typename Point>
using EigenAdaptor = pico_tree::EigenAdaptor<Index, EigenMap<Point>>;

TEST(EigenTest, MetricL1) {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;
  using AdaptorX = EigenAdaptor<PointX>;
  std::vector<PointX, Eigen::aligned_allocator<PointX>> points{{2.0f, 4.0f}};
  AdaptorX adaptor(EigenMap<PointX>(points.data()->data(), 2, 1));
  PointX p{10.0f, 1.0f};

  pico_tree::EigenMetricL1<Scalar> metric(adaptor.sdim());

  EXPECT_FLOAT_EQ(metric(p, adaptor(0)), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(EigenTest, MetricL2) {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;
  using AdaptorX = EigenAdaptor<PointX>;
  std::vector<PointX, Eigen::aligned_allocator<PointX>> points{{2.0f, 4.0f}};
  AdaptorX adaptor(EigenMap<PointX>(points.data()->data(), 2, 1));
  PointX p{10.0f, 1.0f};

  pico_tree::EigenMetricL2<Scalar> metric(adaptor.sdim());

  EXPECT_FLOAT_EQ(metric(p, adaptor(0)), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}

template <typename ColMatrix, typename RowMatrix>
void CheckEigenAdaptorInterface() {
  static_assert(
      !pico_tree::EigenAdaptor<Index, ColMatrix>::RowMajor,
      "ADAPTOR_NOT_COL_MAJOR");
  static_assert(
      pico_tree::EigenAdaptor<Index, RowMatrix>::RowMajor,
      "ADAPTOR_NOT_ROW_MAJOR");

  static_assert(
      pico_tree::EigenAdaptor<Index, ColMatrix>::Dim ==
          ColMatrix::RowsAtCompileTime,
      "ADAPTOR_DIM_NOT_EQUAL_TO_MATRIX_ROWSATCOMPILETIME");

  static_assert(
      pico_tree::EigenAdaptor<Index, RowMatrix>::Dim ==
          ColMatrix::ColsAtCompileTime,
      "ADAPTOR_DIM_NOT_EQUAL_TO_MATRIX_COLSATCOMPILETIME");

  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);

  // Copies of the matrices.
  pico_tree::EigenAdaptor<Index, ColMatrix> col_adaptor(col_matrix);
  pico_tree::EigenAdaptor<Index, RowMatrix> row_adaptor(row_matrix);

  EXPECT_EQ(col_matrix.rows(), col_adaptor.sdim());
  EXPECT_EQ(col_matrix.cols(), col_adaptor.npts());
  EXPECT_EQ(row_matrix.cols(), row_adaptor.sdim());
  EXPECT_EQ(row_matrix.rows(), row_adaptor.npts());

  EXPECT_TRUE(col_adaptor(0).isApprox(col_matrix.col(0)));
  EXPECT_TRUE(col_adaptor(col_adaptor.npts() - 1)
                  .isApprox(col_matrix.col(col_matrix.cols() - 1)));
  EXPECT_TRUE(row_adaptor(0).isApprox(row_matrix.row(0)));
  EXPECT_TRUE(row_adaptor(row_adaptor.npts() - 1)
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

TEST(EigenTest, ValueMove) {
  using ColMatrix = Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor>;
  using RowMatrix = Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::RowMajor>;

  ColMatrix col_matrix = ColMatrix::Random(4, 8);
  RowMatrix row_matrix = RowMatrix::Random(4, 8);

  {
    // Copies of the matrices.
    pico_tree::EigenAdaptor<Index, ColMatrix> col_adaptor(col_matrix);
    pico_tree::EigenAdaptor<Index, RowMatrix> row_adaptor(row_matrix);

    // Proof of copy.
    EXPECT_NE(col_adaptor.matrix().data(), col_matrix.data());
    EXPECT_NE(row_adaptor.matrix().data(), row_matrix.data());
  }

  {
    // The move implementation sets the pointers to nullptr;
    auto col_data = col_matrix.data();
    auto row_data = row_matrix.data();

    pico_tree::EigenAdaptor<Index, ColMatrix> col_adaptor(
        std::move(col_matrix));
    pico_tree::EigenAdaptor<Index, RowMatrix> row_adaptor(
        std::move(row_matrix));

    // Proof of move.
    EXPECT_EQ(col_adaptor.matrix().data(), col_data);
    EXPECT_EQ(row_adaptor.matrix().data(), row_data);
  }
}

TEST(EigenTest, TreeCompatibility) {
  using Adaptor = pico_tree::EigenAdaptor<Index, Eigen::Matrix4Xd>;
  static_assert(
      std::is_same<Index, typename Adaptor::IndexType>::value,
      "ADAPTOR_INDEX_TYPE_INCORRECT");
  using Scalar = typename Adaptor::ScalarType;
  constexpr int Dim = Adaptor::Dim;

  Adaptor adaptor(Eigen::Matrix4Xd::Random(4, 8));
  // Testing of adaptor move compiles.
  pico_tree::KdTree<Index, Scalar, Dim, Adaptor> tree(std::move(adaptor), 10);

  TestKnn(tree, 2);
}
