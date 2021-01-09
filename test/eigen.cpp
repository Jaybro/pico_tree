#include <gtest/gtest.h>

#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>

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
