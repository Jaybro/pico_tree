#include "benchmark.hpp"
#include "nano_adaptor.hpp"

class BmNanoflann : public pico_tree::Benchmark {
 public:
  using NanoAdaptorX = NanoAdaptor<Index, PointX>;
};

template <typename NanoAdaptor>
using NanoKdTreeCt = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<typename NanoAdaptor::ScalarType, NanoAdaptor>,
    NanoAdaptor,
    NanoAdaptor::Dim,
    typename NanoAdaptor::IndexType>;

template <typename NanoAdaptor>
using NanoKdTreeRt = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<typename NanoAdaptor::ScalarType, NanoAdaptor>,
    NanoAdaptor,
    -1,
    typename NanoAdaptor::IndexType>;

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmNanoflann, BuildCt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  NanoAdaptorX adaptor(points_tree_);
  for (auto _ : state) {
    NanoKdTreeCt<NanoAdaptorX> tree(
        PointX::Dim,
        adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(BmNanoflann, BuildRt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  NanoAdaptorX adaptor(points_tree_);
  for (auto _ : state) {
    NanoKdTreeRt<NanoAdaptorX> tree(
        PointX::Dim,
        adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

// Argument 1: Maximum leaf size.
BENCHMARK_REGISTER_F(BmNanoflann, BuildCt)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

BENCHMARK_REGISTER_F(BmNanoflann, BuildRt)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(BmNanoflann, KnnCt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  NanoAdaptorX adaptor(points_tree_);
  NanoKdTreeCt<NanoAdaptorX> tree(
      PointX::Dim,
      adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<Index> indices(knn_count);
    std::vector<Scalar> distances(knn_count);
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      benchmark::DoNotOptimize(
          sum += tree.knnSearch(
              p.data(), knn_count, indices.data(), distances.data()));
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: K nearest neighbors.
BENCHMARK_REGISTER_F(BmNanoflann, KnnCt)
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

BENCHMARK_DEFINE_F(BmNanoflann, RadiusCt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  Scalar radius = static_cast<Scalar>(state.range(1)) / Scalar(10.0);
  Scalar squared = radius * radius;
  NanoAdaptorX adaptor(points_tree_);
  NanoKdTreeCt<NanoAdaptorX> tree(
      PointX::Dim,
      adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<nanoflann::ResultItem<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_test_) {
      benchmark::DoNotOptimize(
          sum += tree.radiusSearch(
              p.data(),
              squared,
              results,
              nanoflann::SearchParameters{0, false}));
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: Search radius (divided by 10.0).
BENCHMARK_REGISTER_F(BmNanoflann, RadiusCt)
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

BENCHMARK_MAIN();
