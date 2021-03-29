#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_understory/cover_tree.hpp>

#include "common.hpp"

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename PointX>
using Space = std::reference_wrapper<std::vector<PointX>>;

template <typename SpaceX>
using Traits = pico_tree::StdTraits<SpaceX>;

template <typename PointX>
using CoverTree = pico_tree::CoverTree<Traits<Space<PointX>>>;

template <typename PointX>
void QueryRadius(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const radius) {
  using Scalar = typename PointX::ScalarType;

  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  CoverTree<PointX> tree(random, Scalar(2.0));

  TestRadius(tree, radius);
}

template <typename PointX>
void QueryKnn(
    int const point_count,
    typename PointX::ScalarType const area_size,
    int const k) {
  using Index = int;
  using Scalar = typename PointX::ScalarType;

  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  CoverTree<PointX> tree(random, Scalar(2.0));

  // This line compile time "tests" the move capability of the tree.
  auto tree2 = std::move(tree);

  TestKnn(tree2, static_cast<Index>(k));
}

}  // namespace

TEST(CoverTreeTest, QueryRadiusSubset2d) {
  QueryRadius<Point2f>(1024 * 128, 100.0f, 2.5f);
}

TEST(CoverTreeTest, QueryKnn1) { QueryKnn<Point2f>(1024 * 128, 100.0f, 1); }

TEST(CoverTreeTest, QueryKnn10) { QueryKnn<Point2f>(1024 * 128, 100.0f, 10); }
