#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/vector_traits.hpp>

#include "common.hpp"

std::vector<Point2f> GetStdVector() { return {{1.0f, 2.0f}}; }

TEST(StdTraitsTest, StdVector) {
  using Space = std::vector<Point2f>;
  using Traits = pico_tree::SpaceTraits<Space>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim>(
      points,
      Point2f::Dim,
      points.size(),
      static_cast<std::size_t>(0),
      points[0].data());
}

TEST(StdTraitsTest, StdRefVector) {
  using Space = std::reference_wrapper<std::vector<Point2f>>;
  using Traits = pico_tree::SpaceTraits<Space>;

  std::vector<Point2f> points = GetStdVector();
  CheckTraits<Traits, Point2f::Dim>(
      std::ref(points),
      Point2f::Dim,
      points.size(),
      static_cast<std::size_t>(0),
      points[0].data());
}
