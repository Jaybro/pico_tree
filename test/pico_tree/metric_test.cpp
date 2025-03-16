#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/point_wrapper.hpp>
#include <pico_tree/metric.hpp>
#include <pico_understory/metric.hpp>

namespace pti = pico_tree::internal;

namespace {

inline constexpr float abs_error = 0.00001f;

}

template <typename Metric_, typename P0_, typename P1_>
inline auto distance(Metric_ const& metric, P0_ const& p0, P1_ const& p1) {
  auto w0 = pti::point_wrapper<P0_>(p0);
  auto w1 = pti::point_wrapper<P0_>(p1);
  return metric(w0.begin(), w0.end(), w1.begin());
}

TEST(MetricTest, AngleDistance) {
  EXPECT_NEAR(pti::s1_distance(0.4f, 0.6f), 0.2f, abs_error);
  EXPECT_NEAR(pti::s1_distance(0.6f, 0.4f), 0.2f, abs_error);
  EXPECT_NEAR(pti::s1_distance(0.1f, 0.9f), 0.2f, abs_error);
  float x = 0.1f;
  float y = 0.9f;
  EXPECT_NEAR(
      pti::squared_s1_distance(x, y),
      pti::s1_distance(x, y) * pti::s1_distance(x, y),
      abs_error);
}

TEST(MetricTest, AngleDistanceBoxEuclidean) {
  EXPECT_NEAR(
      pti::s1_distance_box_euclidean(0.3f, 0.4f, 0.6f), 0.1f, abs_error);
  EXPECT_NEAR(
      pti::s1_distance_box_euclidean(0.7f, 0.4f, 0.6f), 0.1f, abs_error);
  EXPECT_NEAR(
      pti::s1_distance_box_euclidean(0.5f, 0.4f, 0.6f), 0.0f, abs_error);
  EXPECT_NEAR(
      pti::s1_distance_box_euclidean(0.1f, 0.9f, 1.0f), 0.1f, abs_error);
  EXPECT_NEAR(
      pti::squared_s1_distance_box_euclidean(0.1f, 0.9f, 1.0f),
      pti::s1_distance_box_euclidean(0.1f, 0.9f, 1.0f) *
          pti::s1_distance_box_euclidean(0.1f, 0.9f, 1.0f),
      abs_error);
}

TEST(MetricTest, AngleDistanceBoxTopological) {
  EXPECT_NEAR(
      pti::s1_distance_box_topological(0.0f, 0.9f, 0.1f), 0.0f, abs_error);
  EXPECT_NEAR(
      pti::s1_distance_box_topological(0.2f, 0.9f, 0.1f), 0.1f, abs_error);
  EXPECT_NEAR(
      pti::squared_s1_distance_box_topological(0.2f, 0.9f, 0.1f),
      pti::s1_distance_box_topological(0.2f, 0.9f, 0.1f) *
          pti::s1_distance_box_topological(0.2f, 0.9f, 0.1f),
      abs_error);
}

TEST(MetricTest, L1) {
  pico_tree::point_2f p0{2.0f, 4.0f};
  pico_tree::point_2f p1{10.0f, 1.0f};

  pico_tree::metric_l1 metric;

  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2) {
  pico_tree::point_2f p0{7.0f, 5.0f};
  pico_tree::point_2f p1{10.0f, 1.0f};

  pico_tree::metric_l2 metric;

  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  pico_tree::point_2f p0{2.0f, 4.0f};
  pico_tree::point_2f p1{10.0f, 1.0f};

  pico_tree::metric_l2_squared metric;

  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}

TEST(MetricTest, LInf) {
  pico_tree::point_2f p0{2.0f, 4.0f};
  pico_tree::point_2f p1{10.0f, 1.0f};

  pico_tree::metric_linf metric;

  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 8.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, SO2) {
  pico_tree::point_1f p0{0.5f};
  pico_tree::point_1f p1{0.6f};

  pico_tree::metric_so2 metric;

  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 0.1f);
  EXPECT_FLOAT_EQ(metric(-0.1f), 0.1f);
  EXPECT_NEAR(
      metric(0.5f, 0.4f, 0.6f, 0, pico_tree::euclidean_space_tag{}),
      0.0f,
      abs_error);
  EXPECT_NEAR(
      metric(0.2f, 0.9f, 0.1f, 0, pico_tree::topological_space_tag{}),
      0.1f,
      abs_error);
}

TEST(MetricTest, SE2Squared) {
  // These two points are the concatenation of the points of
  // MetricTest.L2Squared and MetricTest.SO2, where the third coordinate is the
  // SO2 angle.
  pico_tree::point_3f p0{2.0f, 4.0f, 0.5f};
  pico_tree::point_3f p1{10.0f, 1.0f, 0.6f};

  pico_tree::metric_se2_squared metric;

  // This is the same as the sum of distances from MetricTest.L2Squared and
  // MetricTest.SO2 (squared).
  EXPECT_FLOAT_EQ(distance(metric, p0, p1), 73.0f + 0.01f);
  EXPECT_FLOAT_EQ(metric(-0.1f), 0.01f);
  EXPECT_NEAR(
      metric(0.5f, 0.4f, 0.6f, 2, pico_tree::euclidean_space_tag{}),
      0.0f,
      abs_error);
  EXPECT_NEAR(
      metric(0.2f, 0.9f, 0.1f, 2, pico_tree::topological_space_tag{}),
      0.01f,
      abs_error);
}
