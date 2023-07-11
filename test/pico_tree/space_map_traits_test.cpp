#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/map_traits.hpp>

#include "common.hpp"

TEST(SpaceMapTraitsTest, PointMap) {
  constexpr pico_tree::Size Dim = 2;
  std::vector<double> scalars = {1.0, 2.0, 3.0, 4.0};
  pico_tree::SpaceMap<pico_tree::PointMap<double, Dim>> map_ct(
      scalars.data(), scalars.size() / Dim);
  pico_tree::SpaceMap<pico_tree::PointMap<double, pico_tree::kDynamicSize>>
      map_rt(scalars.data(), scalars.size() / Dim, Dim);

  CheckSpaceAdaptor<Dim>(
      map_ct,
      map_ct.sdim(),
      map_ct.size(),
      static_cast<pico_tree::Size>(0),
      map_ct[0].data());
  CheckSpaceAdaptor<pico_tree::kDynamicSize>(
      map_rt,
      map_rt.sdim(),
      map_rt.size(),
      static_cast<pico_tree::Size>(0),
      map_rt[0].data());
}

TEST(SpaceMapTraitsTest, Point) {
  std::vector<Point2f> points = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  pico_tree::SpaceMap<Point2f> map(points.data(), points.size());

  CheckSpaceAdaptor<Point2f::Dim>(
      map,
      map.sdim(),
      map.size(),
      static_cast<pico_tree::Size>(0),
      map[0].data());
}
