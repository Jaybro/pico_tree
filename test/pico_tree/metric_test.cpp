#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/metric.hpp>

using PointX = Point2f;
using Index = int;
using Scalar = typename PointX::ScalarType;
static constexpr auto Dim = PointX::Dim;

TEST(MetricTest, L1) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L1<Scalar, Dim> metric(Dim);

  EXPECT_FLOAT_EQ(metric(p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2) {
  PointX p0{7.0f, 5.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2<Scalar, Dim> metric(Dim);

  EXPECT_FLOAT_EQ(metric(p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2Squared<Scalar, Dim> metric(Dim);

  EXPECT_FLOAT_EQ(metric(p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}
