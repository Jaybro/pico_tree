#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/point_wrapper.hpp>
#include <pico_tree/metric.hpp>
#include <pico_understory/metric.hpp>

template <typename Metric_, typename P0, typename P1>
inline auto Distance(Metric_ const& metric, P0 const& p0, P1 const& p1) {
  auto w0 = pico_tree::internal::PointWrapper<P0>(p0);
  auto w1 = pico_tree::internal::PointWrapper<P0>(p1);
  return metric(w0.begin(), w0.end(), w1.begin());
}

TEST(MetricTest, L1) {
  Point2f p0{2.0f, 4.0f};
  Point2f p1{10.0f, 1.0f};

  pico_tree::L1 metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2) {
  Point2f p0{7.0f, 5.0f};
  Point2f p1{10.0f, 1.0f};

  pico_tree::L2 metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  Point2f p0{2.0f, 4.0f};
  Point2f p1{10.0f, 1.0f};

  pico_tree::L2Squared metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}

TEST(MetricTest, LInf) {
  Point2f p0{2.0f, 4.0f};
  Point2f p1{10.0f, 1.0f};

  pico_tree::LInf metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 8.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, SO2) {
  Point1f p0{1.0f};
  Point1f p1{1.1f};

  pico_tree::SO2 metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 0.1f);
  EXPECT_FLOAT_EQ(metric(-0.1f), 0.1f);
  EXPECT_FLOAT_EQ(metric(-0.1f, -0.3f, -0.2f, 0), std::abs(-0.1f - -0.2f));
  EXPECT_FLOAT_EQ(metric(-0.4f, -0.3f, -0.2f, 0), std::abs(-0.4f - -0.3f));
  EXPECT_FLOAT_EQ(metric(-0.25f, -0.3f, -0.2f, 0), 0.0f);
}
