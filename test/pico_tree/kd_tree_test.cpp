#include <gtest/gtest.h>

#include <filesystem>
#include <pico_toolshed/dynamic_space.hpp>
#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

#include "common.hpp"

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename PointX>
using Space = std::reference_wrapper<std::vector<PointX>>;

template <typename PointX>
using KdTree = pico_tree::KdTree<Space<PointX>>;

template <typename PointX>
void QueryRange(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const min_v,
    typename PointX::ScalarType const max_v) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree(random, pico_tree::max_leaf_size_t(8));

  TestBox(tree, min_v, max_v);
}

template <typename PointX>
void QueryRadius(
    int const point_count,
    typename PointX::ScalarType const area_size,
    typename PointX::ScalarType const radius) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree(random, pico_tree::max_leaf_size_t(8));

  TestRadius(tree, radius);
}

template <typename PointX>
void QueryKnn(
    int const point_count,
    typename PointX::ScalarType const area_size,
    int const k) {
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  KdTree<PointX> tree1(random, pico_tree::max_leaf_size_t(8));

  // "Test" move constructor.
  auto tree2 = std::move(tree1);
  // "Test" move assignment.
  tree1 = std::move(tree2);

  TestKnn(tree1, static_cast<typename KdTree<PointX>::IndexType>(k));
}

}  // namespace

TEST(KdTreeTest, QueryRangeSubset2d) {
  QueryRange<Point2f>(1024 * 1024, 100.0f, 15.1f, 34.9f);
}

TEST(KdTreeTest, QueryRangeAll2d) {
  QueryRange<Point2f>(1024, 10.0f, 0.0f, 10.0f);
}

TEST(KdTreeTest, QueryRadiusSubset2d) {
  QueryRadius<Point2f>(1024 * 1024, 100.0f, 2.5f);
}

TEST(KdTreeTest, QueryKnn1) { QueryKnn<Point2f>(1024 * 1024, 100.0f, 1); }

TEST(KdTreeTest, QueryKnn10) { QueryKnn<Point2f>(1024 * 1024, 100.0f, 10); }

TEST(KdTreeTest, QuerySo2Knn4) {
  using PointX = Point1f;
  using SpaceX = Space<PointX>;

  const auto pi = pico_tree::internal::kPi<typename KdTree<PointX>::ScalarType>;
  std::vector<PointX> random = GenerateRandomN<PointX>(256 * 256, -pi, pi);
  pico_tree::KdTree<SpaceX, pico_tree::SO2> tree(
      random, pico_tree::max_leaf_size_t(10));
  TestKnn(tree, static_cast<typename KdTree<PointX>::IndexType>(8), PointX{pi});
}

TEST(KdTreeTest, WriteRead) {
  using Index = int;
  using Scalar = typename Point2f::ScalarType;
  Index point_count = 100;
  Scalar area_size = 2;
  std::vector<Point2f> random =
      GenerateRandomN<Point2f>(point_count, area_size);

  std::string filename = "tree.bin";

  // Compile time known dimensions.
  {
    // The points are not stored.
    KdTree<Point2f> tree(random, pico_tree::max_leaf_size_t(1));
    KdTree<Point2f>::Save(tree, filename);
  }
  {
    // Points are required to load the tree.
    KdTree<Point2f> tree = KdTree<Point2f>::Load(random, filename);
    TestKnn(tree, Index(20));
  }

  EXPECT_TRUE(std::filesystem::remove(filename));

  // Run time known dimensions.
  using DSpace = DynamicSpace<Space<Point2f>>;

  DSpace drandom(random);

  {
    static_assert(
        pico_tree::KdTree<DSpace>::Dim == pico_tree::kDynamicSize,
        "KD_TREE_DIM_NOT_DYNAMIC");
    // The points are not stored.
    pico_tree::KdTree<DSpace> tree(drandom, pico_tree::max_leaf_size_t(1));
    pico_tree::KdTree<DSpace>::Save(tree, filename);
  }
  {
    // Points are required to load the tree.
    pico_tree::KdTree<DSpace> tree =
        pico_tree::KdTree<DSpace>::Load(drandom, filename);
    TestKnn(tree, 20);
  }

  EXPECT_TRUE(std::filesystem::remove(filename));
}
