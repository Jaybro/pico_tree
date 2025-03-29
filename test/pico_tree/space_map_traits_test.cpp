#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/map_traits.hpp>

#include "common.hpp"

TEST(SpaceMapTraitsTest, PointMap) {
  constexpr pico_tree::size_t dim = 2;
  std::vector<double> scalars = {1.0, 2.0, 3.0, 4.0};
  pico_tree::space_map<pico_tree::point_map<double, dim>> map_ct(
      scalars.data(), scalars.size() / dim);
  pico_tree::space_map<pico_tree::point_map<double, pico_tree::dynamic_extent>>
      map_rt(scalars.data(), scalars.size() / dim, dim);

  check_space_adaptor<dim>(
      map_ct,
      map_ct.sdim(),
      map_ct.size(),
      static_cast<pico_tree::size_t>(0),
      map_ct[0].data());
  check_space_adaptor<pico_tree::dynamic_extent>(
      map_rt,
      map_rt.sdim(),
      map_rt.size(),
      static_cast<pico_tree::size_t>(0),
      map_rt[0].data());
}

TEST(SpaceMapTraitsTest, Point) {
  std::vector<pico_tree::point_2f> points = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  pico_tree::space_map<pico_tree::point_2f> map(points.data(), points.size());

  check_space_adaptor<pico_tree::point_2f::dim>(
      map,
      map.sdim(),
      map.size(),
      static_cast<pico_tree::size_t>(0),
      map[0].data());
}
