#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>

#include "common.hpp"

std::vector<Point2f> GetStdVector() { return {{1.0f, 2.0f}}; }

TEST(StdTraitsTest, StdVector) {
  using Space = std::vector<Point2f>;
  using Traits = pico_tree::StdTraits<Space>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim, int>(
      points,
      Point2f::Dim,
      points.size(),
      static_cast<std::size_t>(0),
      points[0].data);
}

TEST(StdTraitsTest, StdRefVector) {
  using Space = std::reference_wrapper<std::vector<Point2f>>;
  using Traits = pico_tree::StdTraits<Space, std::size_t>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim, std::size_t>(
      std::ref(points),
      Point2f::Dim,
      points.size(),
      static_cast<std::size_t>(0),
      points[0].data);
}
