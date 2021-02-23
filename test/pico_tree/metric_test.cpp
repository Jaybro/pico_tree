#include <gtest/gtest.h>

#include <pico_adaptor.hpp>
#include <pico_tree/metric.hpp>

TEST(MetricTest, L1) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using AdaptorX = PicoAdaptor<Index, PointX>;
  constexpr auto Dim = PointX::Dim;
  std::vector<PointX> points{{2.0f, 4.0f}};
  AdaptorX adaptor(points);
  PointX p{10.0f, 1.0f};

  pico_tree::L1<Scalar, Dim> metric(adaptor.sdim());

  EXPECT_FLOAT_EQ(metric(p, adaptor(0)), 11.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 12.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 3.1f);
}

TEST(MetricTest, L2Squared) {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::ScalarType;
  using AdaptorX = PicoAdaptor<Index, PointX>;
  constexpr auto Dim = PointX::Dim;
  std::vector<PointX> points{{2.0f, 4.0f}};
  AdaptorX adaptor(points);
  PointX p{10.0f, 1.0f};

  pico_tree::L2Squared<Scalar, Dim> metric(adaptor.sdim());

  EXPECT_FLOAT_EQ(metric(p, adaptor(0)), 73.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f, 8.9f), 144.0f);
  EXPECT_FLOAT_EQ(metric(-3.1f), 9.61f);
}
