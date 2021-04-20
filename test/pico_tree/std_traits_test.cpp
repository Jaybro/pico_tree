#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>

#include "common.hpp"

std::vector<Point2f> GetStdVector() { return {{1.0f, 2.0f}}; }

template <typename Traits>
void CheckPoint(typename Traits::SpaceType const& points) {
  Point2f const& p = Traits::PointAt(points, 0);
  EXPECT_FLOAT_EQ(p(0), 1.0f);
  EXPECT_FLOAT_EQ(p(1), 2.0f);
}

TEST(StdTraitsTest, StdVector) {
  using Space = std::vector<Point2f>;
  using Traits = pico_tree::StdTraits<Space>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim, int>(points, Point2f::Dim, points.size());
  CheckPoint<Traits>(points);
}

TEST(StdTraitsTest, StdRefVector) {
  using Space = std::reference_wrapper<std::vector<Point2f>>;
  using Traits = pico_tree::StdTraits<Space, std::size_t>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim, std::size_t>(
      std::ref(points), Point2f::Dim, points.size());
  CheckPoint<Traits>(std::ref(points));
}
