#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>

#include "benchmark.hpp"

class BmPicoKdTree : public pico_tree::Benchmark {
 public:
};

// Index explicitly set to int.
template <typename PointX>
using PicoTraits =
    pico_tree::StdTraits<std::reference_wrapper<std::vector<PointX>>, int>;

template <typename PointX>
using PicoKdTreeCtSldMid = pico_tree::KdTree<PicoTraits<PointX>>;

template <typename PointX>
using PicoKdTreeRtSldMid = pico_tree::KdTree<
    PicoTraits<PointX>,
    pico_tree::L2Squared<PicoTraits<PointX>>,
    pico_tree::SplitterSlidingMidpoint<PicoTraits<PointX>>,
    pico_tree::kDynamicDim>;

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeCtSldMid<PointX> tree(points_tree_, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeRtSldMid<PointX> tree(points_tree_, max_leaf_size);
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
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);

  PicoKdTreeCtSldMid<PointX> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      tree.SearchKnn(p, knn_count, &results);
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
  int max_leaf_size = state.range(0);
  Scalar radius = static_cast<Scalar>(state.range(1)) / 10.0;
  Scalar squared = radius * radius;

  PicoKdTreeCtSldMid<PointX> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      tree.SearchRadius(p, squared, &results);
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
  int max_leaf_size = state.range(0);
  Scalar radius = static_cast<Scalar>(state.range(1)) / 10.0;

  PicoKdTreeCtSldMid<PointX> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<Index> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      auto min = p - radius;
      auto max = p + radius;
      tree.SearchBox(min, max, &results);
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
  int max_leaf_size = state.range(0);
  Scalar radius = static_cast<Scalar>(state.range(1)) / 10.0;

  PicoKdTreeRtSldMid<PointX> tree(points_tree_, max_leaf_size);

  for (auto _ : state) {
    std::vector<Index> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      auto min = p - radius;
      auto max = p + radius;
      tree.SearchBox(min, max, &results);
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
