#include <gtest/gtest.h>

#include <pico_toolshed/point.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/cover_tree.hpp>

#include "common.hpp"

// The anonymous namespace gives the function a unique "name" when there is
// another one with the exact same signature.
namespace {

template <typename Point_>
using space = std::reference_wrapper<std::vector<Point_>>;

template <typename Point_>
using cover_tree = pico_tree::cover_tree<space<Point_>>;

template <typename Point_>
void query_radius(
    std::size_t const point_count,
    typename Point_::scalar_type const area_size,
    typename Point_::scalar_type const radius) {
  using scalar_type = typename Point_::scalar_type;

  std::vector<Point_> random =
      pico_tree::generate_random_n<Point_>(point_count, area_size);
  cover_tree<Point_> tree(random, scalar_type(2.0));

  test_radius(tree, radius);
}

template <typename Point_>
void query_knn(
    std::size_t const point_count,
    typename Point_::scalar_type const area_size,
    pico_tree::size_t const k) {
  using scalar_type = typename Point_::scalar_type;

  std::vector<Point_> random =
      pico_tree::generate_random_n<Point_>(point_count, area_size);
  cover_tree<Point_> tree(random, scalar_type(2.0));

  // This line compile time "tests" the move capability of the tree.
  auto tree2 = std::move(tree);

  test_knn(tree2, k);
}

}  // namespace

TEST(CoverTreeTest, QueryRadiusSubset2d) {
  query_radius<pico_tree::point_2f>(1024 * 128, 100.0f, 2.5f);
}

TEST(CoverTreeTest, QueryKnn1) {
  query_knn<pico_tree::point_2f>(1024 * 128, 100.0f, 1);
}

TEST(CoverTreeTest, QueryKnn10) {
  query_knn<pico_tree::point_2f>(1024 * 128, 100.0f, 10);
}
