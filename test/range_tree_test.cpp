#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <pico_adaptor.hpp>
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

template <typename Adaptor1d>
using RangeTree1d = pico_tree::RangeTree1d<
    typename Adaptor1d::Index,
    typename Adaptor1d::Scalar,
    Adaptor1d>;

template <typename Adaptor2d>
using RangeTree2d = pico_tree::RangeTree2d<
    typename Adaptor2d::Index,
    typename Adaptor2d::Scalar,
    Adaptor2d>;

template <typename Adaptor3d>
using RangeTree3d = pico_tree::RangeTree3d<
    typename Adaptor3d::Index,
    typename Adaptor3d::Scalar,
    Adaptor3d>;

using Index = int;

template <typename T>
struct PtsTraits;

template <>
struct PtsTraits<Point1f> {
  using Adaptor = PicoAdaptor<Index, Point1f>;
  using Tree = RangeTree1d<Adaptor>;
};

template <>
struct PtsTraits<Point2f> {
  using Adaptor = PicoAdaptor<Index, Point2f>;
  using Tree = RangeTree2d<Adaptor>;
};

template <>
struct PtsTraits<Point3f> {
  using Adaptor = PicoAdaptor<Index, Point3f>;
  using Tree = RangeTree3d<Adaptor>;
};

template <typename PointX>
using AdaptorXd = typename PtsTraits<PointX>::Adaptor;
template <typename PointX>
using RangeTreeXd = typename PtsTraits<PointX>::Tree;

TEST(RangeTreeTest, RangeTree1d) {
  std::vector<Point1f> raw{{{0}}, {{1}}, {{5}}, {{4}}, {{3}}, {{6}}};
  AdaptorXd<Point1f> adaptor(raw);
  RangeTreeXd<Point1f> rt(adaptor);

  EXPECT_EQ(rt.SearchNearest(4.9f), 2);
  std::vector<int> indices;
  rt.SearchBox(-1.0f, 3.1f, &indices);
  EXPECT_THAT(indices, testing::ElementsAre(0, 1, 4));
}

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename PointX>
void QueryRange(
    int const point_count,
    typename PointX::Scalar const area_size,
    typename PointX::Scalar const min_v,
    typename PointX::Scalar const max_v) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  AdaptorXd<PointX> adaptor(random);
  RangeTreeXd<PointX> tree(adaptor);

  TestRange(tree, min_v, max_v);
}

}  // namespace

TEST(RangeTreeTest, QueryRangeSubset2d) {
  QueryRange<Point2f>(1024 * 1024, 100, 15.1, 34.9);
}

TEST(RangeTreeTest, QueryRangeAll2d) {
  QueryRange<Point2f>(1024, 10.0, 0.0, 10.0);
}

TEST(RangeTreeTest, QueryRangeSubset3d) {
  QueryRange<Point3f>(1024 * 8, 1000, 15.1, 34.9);
}

TEST(RangeTreeTest, QueryRangeAll3d) {
  QueryRange<Point3f>(1024, 10.0, 0.0, 10.0);
}
