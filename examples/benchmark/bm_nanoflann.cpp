#include "benchmark.hpp"
#include "nano_adaptor.hpp"

class BmNanoflann : public pico_tree::Benchmark {
 public:
  using nano_adaptor_type = nano_adaptor<index_type, point_type>;
};

template <typename NanoAdaptor_>
using nano_kd_tree_ct = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::
        L2_Simple_Adaptor<typename NanoAdaptor_::scalar_type, NanoAdaptor_>,
    NanoAdaptor_,
    NanoAdaptor_::dim,
    typename NanoAdaptor_::index_type>;

template <typename NanoAdaptor_>
using nano_kd_tree_rt = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::
        L2_Simple_Adaptor<typename NanoAdaptor_::scalar_type, NanoAdaptor_>,
    NanoAdaptor_,
    -1,
    typename NanoAdaptor_::index_type>;

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmNanoflann, BuildCt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  nano_adaptor_type adaptor(points_tree_);
  for (auto _ : state) {
    nano_kd_tree_ct<nano_adaptor_type> tree(
        point_type::dim,
        adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(BmNanoflann, BuildRt)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  nano_adaptor_type adaptor(points_tree_);
  for (auto _ : state) {
    nano_kd_tree_rt<nano_adaptor_type> tree(
        point_type::dim,
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
  nano_adaptor_type adaptor(points_tree_);
  nano_kd_tree_ct<nano_adaptor_type> tree(
      point_type::dim,
      adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<index_type> indices(knn_count);
    std::vector<scalar_type> distances(knn_count);
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
  scalar_type radius =
      static_cast<scalar_type>(state.range(1)) / scalar_type(10.0);
  scalar_type squared = radius * radius;
  nano_adaptor_type adaptor(points_tree_);
  nano_kd_tree_ct<nano_adaptor_type> tree(
      point_type::dim,
      adaptor,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<nanoflann::ResultItem<index_type, scalar_type>> results;
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
