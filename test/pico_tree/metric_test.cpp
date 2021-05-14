#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/metric.hpp>
#include <pico_understory/metric.hpp>

using PointX = Point2f;
using SpaceX = std::vector<PointX>;
using TraitsX = pico_tree::StdTraits<SpaceX>;

TEST(MetricTest, L1) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L1<TraitsX> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2) {
  PointX p0{7.0f, 5.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2<TraitsX> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2Squared<TraitsX> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}

TEST(MetricTest, SO2) {
  PointX p0{1.0f};
  PointX p1{1.1f};

  pico_tree::SO2<TraitsX> metric;

  EXPECT_FLOAT_EQ(metric(p0, p1), 0.1f);
  EXPECT_FLOAT_EQ(metric(-0.1f), 0.1f);
  EXPECT_TRUE(metric(-3.14f, 3.14f, 0) < 0.1f);
  EXPECT_FLOAT_EQ(metric(-0.1f, -0.3f, -0.2f, 0), metric(-0.1f, -0.2f, 0));
  EXPECT_FLOAT_EQ(metric(-0.4f, -0.3f, -0.2f, 0), metric(-0.4f, -0.3f, 0));
  EXPECT_FLOAT_EQ(metric(-0.25f, -0.3f, -0.2f, 0), 0.0f);
}
