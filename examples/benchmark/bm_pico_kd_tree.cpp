#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>

#include "benchmark.hpp"

template <int Dims, typename PicoAdaptor>
using SplitterLongestMedian = pico_tree::SplitterLongestMedian<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeCtSldMid = pico_tree::KdTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    PicoAdaptor::Dim,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeCtLngMed = pico_tree::KdTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    PicoAdaptor::Dim,
    PicoAdaptor,
    pico_tree::L2Squared<typename PicoAdaptor::ScalarType, PicoAdaptor::Dim>,
    SplitterLongestMedian<PicoAdaptor::Dim, PicoAdaptor>>;

template <typename PicoAdaptor>
using PicoKdTreeRtSldMid = pico_tree::KdTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    pico_tree::kDynamicDim,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeRtLngMed = pico_tree::KdTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    pico_tree::kDynamicDim,
    PicoAdaptor,
    pico_tree::L2Squared<typename PicoAdaptor::ScalarType, PicoAdaptor::Dim>,
    SplitterLongestMedian<pico_tree::kDynamicDim, PicoAdaptor>>;

class BmPicoKdTree : public pico_tree::Benchmark {
 public:
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtSldMid)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildCtLngMed)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeCtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtSldMid)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeRtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, BuildRtLngMed)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeRtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);
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
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);

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
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);

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
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);

  for (auto _ : state) {
    pico_tree::Neighbor<Index, Scalar> result;
    for (auto const& p : points_) {
      tree.SearchNn(p, &result);
    }
  }
}

BENCHMARK_DEFINE_F(BmPicoKdTree, NnCtLngMed)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);

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

BENCHMARK_DEFINE_F(BmPicoKdTree, RadiusCtSldMid)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 4.0;
  double squared = radius * radius;
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);

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
// Argument 2: Search radius (divided by 4.0).
BENCHMARK_REGISTER_F(BmPicoKdTree, RadiusCtSldMid)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 1})
    ->Args({6, 1})
    ->Args({8, 1})
    ->Args({10, 1})
    ->Args({12, 1})
    ->Args({14, 1})
    ->Args({1, 2})
    ->Args({6, 2})
    ->Args({8, 2})
    ->Args({10, 2})
    ->Args({12, 2})
    ->Args({14, 2});

BENCHMARK_MAIN();
