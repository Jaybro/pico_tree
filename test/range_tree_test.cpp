#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <pico_point_set.hpp>
#include <pico_tree/range_tree.hpp>
#include <scoped_timer.hpp>

#include "common.hpp"

// TODO Experimental
namespace pico_tree {

template <typename Index, typename Scalar, typename Points>
class RangeTree3d : public internal::RangeTreeNd_<Index, Scalar, 0, Points> {
 public:
  explicit RangeTree3d(Points const& points)
      : internal::RangeTreeNd_<Index, Scalar, 0, Points>(points) {}
};

}  // namespace pico_tree

template <typename PointSet1d>
using RangeTree1d = pico_tree::RangeTree1d<
    typename PointSet1d::Index,
    typename PointSet1d::Scalar,
    PointSet1d>;

template <typename PointSet2d>
using RangeTree2d = pico_tree::RangeTree2d<
    typename PointSet2d::Index,
    typename PointSet2d::Scalar,
    PointSet2d>;

template <typename PointSet3d>
using RangeTree3d = pico_tree::RangeTree3d<
    typename PointSet3d::Index,
    typename PointSet3d::Scalar,
    PointSet3d>;

using Index = int;

template <typename T>
struct PtsTraits;

template <>
struct PtsTraits<Point1f> {
  using PointSet = PicoPointSet<Index, Point1f>;
  using Tree = RangeTree1d<PointSet>;
};

template <>
struct PtsTraits<Point2f> {
  using PointSet = PicoPointSet<Index, Point2f>;
  using Tree = RangeTree2d<PointSet>;
};

template <>
struct PtsTraits<Point3f> {
  using PointSet = PicoPointSet<Index, Point3f>;
  using Tree = RangeTree3d<PointSet>;
};

template <typename PointsX>
using PicoPointSetXd = typename PtsTraits<PointsX>::PointSet;
template <typename PointsX>
using PicoRangeTreeXd = typename PtsTraits<PointsX>::Tree;

TEST(RangeTreeTest, RangeTree1d) {
  std::vector<Point1f> raw{{{0}}, {{1}}, {{5}}, {{4}}, {{3}}, {{6}}};
  PicoPointSetXd<Point1f> points(raw);
  PicoRangeTreeXd<Point1f> rt(points);

  EXPECT_EQ(rt.SearchNearest(4.9f), 2);
  std::vector<int> indices;
  rt.SearchRange(-1.0f, 3.1f, &indices);
  EXPECT_THAT(indices, testing::ElementsAre(0, 1, 4));
}

template <typename PointX>
void QueryRange(
    int const point_count,
    typename PointX::Scalar const area_size,
    typename PointX::Scalar const min_v,
    typename PointX::Scalar const max_v) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoPointSetXd<PointX> points(random);
  PicoRangeTreeXd<PointX> tree(points);

  TestRange(tree, min_v, max_v);
}

TEST(RangeTreeTest, QuerySubset2d) {
  QueryRange<Point2f>(1024 * 1024, 100, 15.1, 34.9);
}

TEST(RangeTreeTest, QueryAll2d) { QueryRange<Point2f>(1024, 10.0, 0.0, 10.0); }

TEST(RangeTreeTest, DISABLED_QuerySubset3d) {
  QueryRange<Point3f>(1024 * 8, 1000, 15.1, 34.9);
}

TEST(RangeTreeTest, DISABLED_QueryAll3d) {
  QueryRange<Point3f>(1024, 10.0, 0.0, 10.0);
}
