#include <pico_toolshed/dynamic_space.hpp>
#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

#include "benchmark.hpp"

class BmPicoKdTree : public pico_tree::Benchmark {
 public:
};

template <typename Point_>
using pico_ct_space = std::reference_wrapper<std::vector<Point_>>;

template <typename Point_>
using pico_rt_space =
    pico_tree::dynamic_space<std::reference_wrapper<std::vector<Point_>>>;

template <typename Point_>
using pico_kd_tree_ct_sld_mid = pico_tree::kd_tree<pico_ct_space<Point_>>;

template <typename Point_>
using pico_kd_tree_rt_sld_mid = pico_tree::kd_tree<pico_rt_space<Point_>>;

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);

  for (auto _ : state) {
    pico_kd_tree_ct_sld_mid<point_type> tree(points_tree_, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);

  for (auto _ : state) {
    pico_kd_tree_rt_sld_mid<point_type> tree(
        pico_rt_space<point_type>(points_tree_), max_leaf_size);
  }
}

// Argument 1: Maximum leaf size.
BENCHMARK_REGISTER_F(BmPicoKdTree, BuildCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

BENCHMARK_REGISTER_F(BmPicoKdTree, BuildRtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, KnnCtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);
  std::size_t knn_count = state.range(1);

  pico_kd_tree_ct_sld_mid<point_type> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::neighbor<index_type, scalar_type>> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      tree.search_knn(p, knn_count, results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

BENCHMARK_REGISTER_F(BmPicoKdTree, KnnCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 1})
    ->Args({6, 1})
    ->Args({8, 1})
    ->Args({10, 1})
    ->Args({12, 1})
    ->Args({14, 1})
    ->Args({1, 4})
    ->Args({6, 4})
    ->Args({8, 4})
    ->Args({10, 4})
    ->Args({12, 4})
    ->Args({14, 4})
    ->Args({1, 8})
    ->Args({6, 8})
    ->Args({8, 8})
    ->Args({10, 8})
    ->Args({12, 8})
    ->Args({14, 8})
    ->Args({1, 12})
    ->Args({6, 12})
    ->Args({8, 12})
    ->Args({10, 12})
    ->Args({12, 12})
    ->Args({14, 12});

// ****************************************************************************
// Radius
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, RadiusCtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);
  scalar_type radius =
      static_cast<scalar_type>(state.range(1)) / scalar_type(10.0);
  scalar_type squared = radius * radius;

  pico_kd_tree_ct_sld_mid<point_type> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::neighbor<index_type, scalar_type>> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      tree.search_radius(p, squared, results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: Search radius (divided by 10.0).
BENCHMARK_REGISTER_F(BmPicoKdTree, RadiusCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 15})
    ->Args({6, 15})
    ->Args({8, 15})
    ->Args({10, 15})
    ->Args({12, 15})
    ->Args({14, 15})
    ->Args({1, 30})
    ->Args({6, 30})
    ->Args({8, 30})
    ->Args({10, 30})
    ->Args({12, 30})
    ->Args({14, 30});

// ****************************************************************************
// Box
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, BoxCtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);
  scalar_type radius =
      static_cast<scalar_type>(state.range(1)) / scalar_type(10.0);

  pico_kd_tree_ct_sld_mid<point_type> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<index_type> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      auto min = p - radius;
      auto max = p + radius;
      tree.search_box(min, max, results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: Search radius (half the width of the box divided by 10.0).
BENCHMARK_REGISTER_F(BmPicoKdTree, BoxCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 15})
    ->Args({6, 15})
    ->Args({8, 15})
    ->Args({10, 15})
    ->Args({12, 15})
    ->Args({14, 15});

BENCHMARK_DEFINE_F(BmPicoKdTree, BoxRtSldMid)(benchmark::State& state) {
  pico_tree::max_leaf_size_t max_leaf_size = state.range(0);
  scalar_type radius =
      static_cast<scalar_type>(state.range(1)) / scalar_type(10.0);

  pico_kd_tree_rt_sld_mid<point_type> tree(
      pico_rt_space<point_type>(points_tree_), max_leaf_size);

  for (auto _ : state) {
    std::vector<index_type> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      auto min = p - radius;
      auto max = p + radius;
      tree.search_box(min, max, results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// The run-time version of the box-search is interesting to test because it
// internally maintains bounding boxes that are represented differently
// depending whether we know the dimension of the space at compile-time or
// run-time.
// Argument 1: Maximum leaf size.
// Argument 2: Search radius (half the width of the box divided by 10.0).
BENCHMARK_REGISTER_F(BmPicoKdTree, BoxRtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 15})
    ->Args({6, 15})
    ->Args({8, 15})
    ->Args({10, 15})
    ->Args({12, 15})
    ->Args({14, 15});

BENCHMARK_MAIN();
