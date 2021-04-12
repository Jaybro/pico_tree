#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>

#include "benchmark.hpp"

// Index explicitly set to int.
template <typename PointX>
using PicoTraits =
    pico_tree::StdTraits<std::reference_wrapper<std::vector<PointX>>, int>;

template <typename PointX>
using PicoKdTreeCtSldMid = pico_tree::KdTree<PicoTraits<PointX>>;

template <typename PointX>
using PicoKdTreeCtLngMed = pico_tree::KdTree<
    PicoTraits<PointX>,
    pico_tree::L2Squared<PicoTraits<PointX>>,
    pico_tree::SplitterLongestMedian<PicoTraits<PointX>>>;

template <typename PointX>
using PicoKdTreeRtSldMid = pico_tree::KdTree<
    PicoTraits<PointX>,
    pico_tree::L2Squared<PicoTraits<PointX>>,
    pico_tree::SplitterSlidingMidpoint<PicoTraits<PointX>>,
    pico_tree::kDynamicDim>;

template <typename PointX>
using PicoKdTreeRtLngMed = pico_tree::KdTree<
    PicoTraits<PointX>,
    pico_tree::L2Squared<PicoTraits<PointX>>,
    pico_tree::SplitterLongestMedian<PicoTraits<PointX>>,
    pico_tree::kDynamicDim>;

class BmPicoKdTree : public pico_tree::Benchmark {
 public:
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeCtSldMid<PointX> tree(points_, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtLngMed)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeCtLngMed<PointX> tree(points_, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeRtSldMid<PointX> tree(points_, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtLngMed)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  for (auto _ : state) {
    PicoKdTreeRtLngMed<PointX> tree(points_, max_leaf_size);
  }
}

// Argument 1: Maximum leaf size.
BENCHMARK_REGISTER_F(BmPicoKdTree, BuildCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(BmPicoKdTree, BuildCtLngMed)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

BENCHMARK_REGISTER_F(BmPicoKdTree, BuildRtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(BmPicoKdTree, BuildRtLngMed)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, KnnCtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);

  PicoKdTreeCtSldMid<PointX> tree(points_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchKnn(p, knn_count, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, KnnCtLngMed)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);

  PicoKdTreeCtLngMed<PointX> tree(points_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchKnn(p, knn_count, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, NnCtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  PicoKdTreeCtSldMid<PointX> tree(points_, max_leaf_size);

  for (auto _ : state) {
    pico_tree::Neighbor<Index, Scalar> result;
    for (auto const& p : points_) {
      tree.SearchNn(p, &result);
    }
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, NnCtLngMed)(benchmark::State& state) {
  int max_leaf_size = state.range(0);

  PicoKdTreeCtLngMed<PointX> tree(points_, max_leaf_size);

  for (auto _ : state) {
    pico_tree::Neighbor<Index, Scalar> result;
    for (auto const& p : points_) {
      tree.SearchNn(p, &result);
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

// For a single nn this method performs the fasted. Increasing the amount of nns
// will land it in between sliding midpoint and nanoflann, until it seems to
// become slower than both other methods.
BENCHMARK_REGISTER_F(BmPicoKdTree, KnnCtLngMed)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 4})
    ->Args({6, 4})
    ->Args({8, 4})
    ->Args({10, 4})
    ->Args({12, 4})
    ->Args({14, 4});

// The 2nd argument is used to keep plot_benchmarks.py the same.
BENCHMARK_REGISTER_F(BmPicoKdTree, NnCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 1})
    ->Args({6, 1})
    ->Args({8, 1})
    ->Args({10, 1})
    ->Args({12, 1})
    ->Args({14, 1});

// The 2nd argument is used to keep plot_benchmarks.py the same.
BENCHMARK_REGISTER_F(BmPicoKdTree, NnCtLngMed)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 1})
    ->Args({6, 1})
    ->Args({8, 1})
    ->Args({10, 1})
    ->Args({12, 1})
    ->Args({14, 1});

// ****************************************************************************
// Radius
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, RadiusCtSldMid)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 10.0;
  double squared = radius * radius;

  PicoKdTreeCtSldMid<PointX> tree(points_, max_leaf_size);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchRadius(p, squared, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: Search radius (divided by 10.0).
BENCHMARK_REGISTER_F(BmPicoKdTree, RadiusCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 2})
    ->Args({6, 2})
    ->Args({8, 2})
    ->Args({10, 2})
    ->Args({12, 2})
    ->Args({14, 2})
    ->Args({1, 4})
    ->Args({6, 4})
    ->Args({8, 4})
    ->Args({10, 4})
    ->Args({12, 4})
    ->Args({14, 4});

BENCHMARK_MAIN();
