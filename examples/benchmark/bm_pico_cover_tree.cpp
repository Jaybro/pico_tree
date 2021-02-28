#include <pico_adaptor.hpp>
#include <pico_tree/cover_tree.hpp>
#include <scoped_timer.hpp>

#include "benchmark.hpp"

template <typename PicoAdaptor>
using PicoCoverTree = pico_tree::CoverTree<
    typename PicoAdaptor::IndexType,
    typename PicoAdaptor::ScalarType,
    PicoAdaptor::Dim,
    PicoAdaptor>;

class BmPicoCoverTree : public pico_tree::Benchmark {
 public:
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoCoverTree, BuildCt)(benchmark::State& state) {
  Scalar base = static_cast<Scalar>(state.range(0)) / Scalar(10.0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoCoverTree<PicoAdaptorX> tree(adaptor, base);
  }
}

BENCHMARK_REGISTER_F(BmPicoCoverTree, BuildCt)
    ->Unit(benchmark::kMillisecond)
    ->Arg(13)
    ->DenseRange(14, 20, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoCoverTree, KnnCt)(benchmark::State& state) {
  Scalar base = static_cast<Scalar>(state.range(0)) / Scalar(10.0);
  int knn_count = state.range(1);
  PicoAdaptorX adaptor(points_);
  PicoCoverTree<PicoAdaptorX> tree(adaptor, base);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    std::size_t group = 1000;

    for (std::size_t pi = 0; pi < points_.size(); ++pi) {
      std::size_t group_end = std::min(pi + group, points_.size());
      std::flush(std::cout);
      ScopedTimer timer("query_group");
      for (; pi < group_end; ++pi) {
        auto const& p = points_[pi];
        tree.SearchKnn(p, knn_count, &results);
        benchmark::DoNotOptimize(sum += results.size());
      }
    }
  }
}

BENCHMARK_REGISTER_F(BmPicoCoverTree, KnnCt)
    ->Unit(benchmark::kMillisecond)
    ->Args({13, 1});

BENCHMARK_MAIN();
