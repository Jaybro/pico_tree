#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/vector_traits.hpp>

#include "common.hpp"

TEST(VectorTraitsTest, Interface) {
  std::vector<pico_tree::point_2f> points = {{1.0f, 2.0f}, {3.0f, 4.0f}};

  check_space_adaptor<pico_tree::point_2f::dim>(
      points,
      pico_tree::point_2f::dim,
      points.size(),
      static_cast<pico_tree::size_t>(0),
      points[0].data());
  // VectorTraitsTest is used to test the default std::reference_wrapper<>
  // specialization.
  check_space_adaptor<pico_tree::point_2f::dim>(
      std::ref(points),
      pico_tree::point_2f::dim,
      points.size(),
      static_cast<pico_tree::size_t>(0),
      points[0].data());
  // VectorTraitsTest is used to test the default std::reference_wrapper<const>
  // specialization.
  check_space_adaptor<pico_tree::point_2f::dim>(
      std::cref(points),
      pico_tree::point_2f::dim,
      points.size(),
      static_cast<pico_tree::size_t>(0),
      points[0].data());
}
