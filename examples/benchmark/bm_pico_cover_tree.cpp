#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_understory/cover_tree.hpp>

#include "benchmark.hpp"

// Index explicitly set to int.
template <typename PointX>
using PicoTraits =
    pico_tree::StdTraits<std::reference_wrapper<std::vector<PointX>>, int>;

template <typename PointX>
using PicoCoverTree = pico_tree::CoverTree<PicoTraits<PointX>>;

class BmPicoCoverTree : public pico_tree::Benchmark {
 public:
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoCoverTree, BuildCt)(benchmark::State& state) {
  Scalar base = static_cast<Scalar>(state.range(0)) / Scalar(10.0);

  for (auto _ : state) {
    PicoCoverTree<PointX> tree(points_tree_, base);
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

  PicoCoverTree<PointX> tree(points_tree_, base);

  for (auto _ : state) {
    std::vector<pico_tree::Neighbor<Index, Scalar>> results;
    std::size_t sum = 0;
    std::size_t group = 4000;
    std::size_t pi = 0;
    while (pi < points_test_.size()) {
      std::size_t group_end = std::min(pi + group, points_test_.size());
      std::flush(std::cout);
      ScopedTimer timer("query_group");
      for (; pi < group_end; ++pi) {
        auto const& p = points_test_[pi];
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
