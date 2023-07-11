#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/vector_traits.hpp>

#include "common.hpp"

TEST(VectorTraitsTest, Interface) {
  std::vector<Point2f> points = {{1.0f, 2.0f}, {3.0f, 4.0f}};

  CheckSpaceAdaptor<Point2f::Dim>(
      points,
      Point2f::Dim,
      points.size(),
      static_cast<pico_tree::Size>(0),
      points[0].data());
  // VectorTraitsTest is used to test the default std::reference_wrapper<>
  // specialization.
  CheckSpaceAdaptor<Point2f::Dim>(
      std::ref(points),
      Point2f::Dim,
      points.size(),
      static_cast<pico_tree::Size>(0),
      points[0].data());
  // VectorTraitsTest is used to test the default std::reference_wrapper<const>
  // specialization.
  CheckSpaceAdaptor<Point2f::Dim>(
      std::cref(points),
      Point2f::Dim,
      points.size(),
      static_cast<pico_tree::Size>(0),
      points[0].data());
}
