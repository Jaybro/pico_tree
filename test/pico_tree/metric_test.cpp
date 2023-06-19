#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/metric.hpp>
#include <pico_understory/metric.hpp>

using PointX = Point2f;

template <typename Traits_>
constexpr pico_tree::Size Dimension(typename Traits_::PointType const& point) {
  if constexpr (Traits_::Dim != pico_tree::kDynamicSize) {
    return Traits_::Dim;
  } else {
    return Traits_::Sdim(point);
  }
}

template <typename Metric_, typename P0, typename P1>
inline auto Distance(Metric_ const& metric, P0 const& p0, P1 const& p1) {
  auto c0 = pico_tree::PointTraits<P0>::Coords(p0);
  auto c1 = pico_tree::PointTraits<P1>::Coords(p1);
  return metric(c0, c0 + Dimension<pico_tree::PointTraits<P0>>(p0), c1);
}

TEST(MetricTest, L1) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L1 metric;
  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2) {
  PointX p0{7.0f, 5.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2 metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 5.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  PointX p0{2.0f, 4.0f};
  PointX p1{10.0f, 1.0f};

  pico_tree::L2Squared metric;

  EXPECT_FLOAT_EQ(Distance(metric, p0, p1), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
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
