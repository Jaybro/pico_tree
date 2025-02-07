#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/cover_tree.hpp>

#include "benchmark.hpp"

class BmPicoCoverTree : public pico_tree::Benchmark {
 public:
};

// Index explicitly set to int.
template <typename Point_>
using pico_space = std::reference_wrapper<std::vector<Point_>>;

template <typename Point_>
using pico_cover_tree = pico_tree::cover_tree<pico_space<Point_>>;

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(BmPicoCoverTree, BuildCt)(benchmark::State& state) {
  scalar_type base =
      static_cast<scalar_type>(state.range(0)) / scalar_type(10.0);

  for (auto _ : state) {
    pico_cover_tree<point_type> tree(points_tree_, base);
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
  scalar_type base =
      static_cast<scalar_type>(state.range(0)) / scalar_type(10.0);
  int knn_count = state.range(1);

  pico_cover_tree<point_type> tree(points_tree_, base);

  for (auto _ : state) {
    std::vector<pico_tree::neighbor<index_type, scalar_type>> results;
    std::size_t sum = 0;
    std::size_t group = 4000;
    std::size_t pi = 0;
    while (pi < points_test_.size()) {
      std::size_t group_end = std::min(pi + group, points_test_.size());
      std::flush(std::cout);
      pico_tree::scoped_timer timer("query_group");
      for (; pi < group_end; ++pi) {
        auto const& p = points_test_[pi];
        tree.search_knn(p, knn_count, results);
        benchmark::DoNotOptimize(sum += results.size());
      }
    }
  }
}

BENCHMARK_REGISTER_F(BmPicoCoverTree, KnnCt)
    ->Unit(benchmark::kMillisecond)
    ->Args({13, 1});

BENCHMARK_MAIN();
