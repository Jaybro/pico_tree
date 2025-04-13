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

template <typename Point_>
using space = std::reference_wrapper<std::vector<Point_>>;

template <typename Point_>
using kd_tree = pico_tree::kd_tree<space<Point_>>;

template <typename Point_>
void query_range(
    std::size_t const point_count,
    typename Point_::scalar_type const area_size,
    typename Point_::scalar_type const min_v,
    typename Point_::scalar_type const max_v) {
  std::vector<Point_> random =
      pico_tree::generate_random_n<Point_>(point_count, area_size);
  kd_tree<Point_> tree(random, pico_tree::max_leaf_size_t(8));

  test_box(tree, min_v, max_v);
}

template <typename Point_>
void query_radius(
    std::size_t const point_count,
    typename Point_::scalar_type const area_size,
    typename Point_::scalar_type const radius) {
  std::vector<Point_> random =
      pico_tree::generate_random_n<Point_>(point_count, area_size);
  kd_tree<Point_> tree(random, pico_tree::max_leaf_size_t(8));

  test_radius(tree, radius);
}

template <typename Point_>
void query_knn(
    std::size_t const point_count,
    typename Point_::scalar_type const area_size,
    pico_tree::size_t k) {
  std::vector<Point_> random =
      pico_tree::generate_random_n<Point_>(point_count, area_size);
  kd_tree<Point_> tree1(random, pico_tree::max_leaf_size_t(8));

  // "Test" move constructor.
  auto tree2 = std::move(tree1);
  // "Test" move assignment.
  tree1 = std::move(tree2);

  test_knn(tree1, k);
}

}  // namespace

TEST(KdTreeTest, QueryRangeSubset2d) {
  query_range<pico_tree::point_2f>(1024 * 1024, 100.0f, 15.1f, 34.9f);
}

TEST(KdTreeTest, QueryRangeAll2d) {
  query_range<pico_tree::point_2f>(1024, 10.0f, 0.0f, 10.0f);
}

TEST(KdTreeTest, QueryRadiusSubset2d) {
  query_radius<pico_tree::point_2f>(1024 * 1024, 100.0f, 2.5f);
}

TEST(KdTreeTest, QueryKnn1) {
  query_knn<pico_tree::point_2f>(1024 * 1024, 100.0f, 1);
}

TEST(KdTreeTest, QueryKnn10) {
  query_knn<pico_tree::point_2f>(1024 * 1024, 100.0f, 10);
}

TEST(KdTreeTest, QuerySo2Knn4) {
  using point_type = pico_tree::point_1f;
  using scalar_type = typename point_type::scalar_type;
  using space_type = space<point_type>;

  std::vector<point_type> random = pico_tree::generate_random_n<point_type>(
      256 * 256, scalar_type(0.0), scalar_type(1.0));
  pico_tree::kd_tree<space_type, pico_tree::metric_so2> tree(
      random, pico_tree::max_leaf_size_t(10));

  test_knn(tree, 8, point_type{1.0});
  test_box(tree, scalar_type(0.90), scalar_type(1.00));
  test_box(tree, scalar_type(0.95), scalar_type(0.05));
}

TEST(KdTreeTest, WriteRead) {
  using point_type = pico_tree::point_2f;
  using index_type = int;
  using scalar_type = typename point_type::scalar_type;
  std::size_t point_count = 100;
  scalar_type area_size = 2;
  std::vector<point_type> random =
      pico_tree::generate_random_n<point_type>(point_count, area_size);

  std::string filename = "tree.bin";

  // Compile time known dimensions.
  {
    // The points are not stored.
    kd_tree<point_type> tree(random, pico_tree::max_leaf_size_t(1));
    kd_tree<point_type>::save(tree, filename);
  }
  {
    // Points are required to load the tree.
    kd_tree<point_type> tree = kd_tree<point_type>::load(random, filename);
    test_knn(tree, index_type(20));
  }

  EXPECT_TRUE(std::filesystem::remove(filename));

  // Run time known dimensions.
  using space_type = pico_tree::dynamic_space<space<point_type>>;

  space_type drandom(random);

  {
    static_assert(
        pico_tree::kd_tree<space_type>::dim == pico_tree::dynamic_extent,
        "KD_TREE_DIM_NOT_DYNAMIC");
    // The points are not stored.
    pico_tree::kd_tree<space_type> tree(drandom, pico_tree::max_leaf_size_t(1));
    pico_tree::kd_tree<space_type>::save(tree, filename);
  }
  {
    // Points are required to load the tree.
    pico_tree::kd_tree<space_type> tree =
        pico_tree::kd_tree<space_type>::load(drandom, filename);
    test_knn(tree, 20);
  }

  EXPECT_TRUE(std::filesystem::remove(filename));
}

TEST(KdTreeTest, LeafRanges) {
  using point_type = pico_tree::point_2f;
  using scalar_type = typename point_type::scalar_type;
  using index_type = int;
  std::size_t point_count = 100;
  scalar_type area_size = 2;
  std::vector<point_type> random =
      pico_tree::generate_random_n<point_type>(point_count, area_size);

  kd_tree<point_type> tree(random, pico_tree::max_leaf_depth_t(2));

  auto leaf_ranges = tree.leaf_ranges();
  EXPECT_EQ(leaf_ranges.size(), 4);
  if (!leaf_ranges.empty()) {
    std::ptrdiff_t sum_range_point_count = 0;
    index_type index_sum = 0;
    for (auto const& r : leaf_ranges) {
      sum_range_point_count += std::distance(r.begin(), r.end());
      for (int i : r) {
        index_sum += i;
      }
    }
    EXPECT_EQ(static_cast<std::size_t>(sum_range_point_count), point_count);
    EXPECT_EQ(
        static_cast<std::size_t>(index_sum),
        (point_count - 1) * point_count / 2);
    EXPECT_EQ(
        std::distance(leaf_ranges.front().begin(), leaf_ranges.back().end()),
        point_count);
  }
}
