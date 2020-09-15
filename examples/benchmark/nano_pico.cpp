#include <benchmark/benchmark.h>

#include <pico_point_set.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

#include "format_bin.hpp"
#include "nano_point_set.hpp"

class KdTreeBenchmark : public benchmark::Fixture {
 protected:
  using Index = int;
  using Scalar = double;
  using PointX = Point3d;
  using NanoPointSetX = NanoPointSet<Index, PointX>;
  using PicoPointSetX = PicoPointSet<Index, PointX>;

 public:
  KdTreeBenchmark() {
    // Here you may need to be patient. Depending on the size of the compiled
    // binary.
    pico_tree::ReadBin("./scans.bin", &points_);
  }

 protected:
  std::vector<PointX> points_;
};

// ****************************************************************************
// Building the tree
// ****************************************************************************

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtNanoBuildTree)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  NanoPointSetX nano_set(points_);
  for (auto _ : state) {
    NanoflannKdTree<NanoPointSetX> tree(
        PointX::Dims,
        nano_set,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtPicoBuildTree)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoPointSetX pico_set(points_);
  for (auto _ : state) {
    KdTree<PicoPointSetX> tree(pico_set, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, RtNanoBuildTree)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  NanoPointSetX nano_set(points_);
  for (auto _ : state) {
    NanoflannKdTreeRt<NanoPointSetX> tree(
        PointX::Dims,
        nano_set,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, RtPicoBuildTree)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoPointSetX pico_set(points_);
  for (auto _ : state) {
    KdTreeRt<PicoPointSetX> tree(pico_set, max_leaf_size);
  }
}

// Argument 1: Maximum leaf size.
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtNanoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

BENCHMARK_REGISTER_F(KdTreeBenchmark, RtNanoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, RtPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtNanoKnn)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  NanoPointSetX nano_set(points_);
  NanoflannKdTree<NanoPointSetX> tree(
      PointX::Dims,
      nano_set,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<Index> indices(knn_count);
    std::vector<Scalar> distances(knn_count);
    std::size_t sum = 0;
    for (auto const& p : points_) {
      benchmark::DoNotOptimize(
          sum +=
          tree.knnSearch(p.data, knn_count, indices.data(), distances.data()));
    }
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtPicoKnn)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  PicoPointSetX pico_set(points_);
  KdTree<PicoPointSetX> tree(pico_set, max_leaf_size);

  for (auto _ : state) {
    std::vector<std::pair<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchKnn(p, knn_count, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: K nearest neighbors.
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtNanoKnn)
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

BENCHMARK_REGISTER_F(KdTreeBenchmark, CtPicoKnn)
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

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtNanoRadius)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 4.0;
  double squared = radius * radius;
  NanoPointSetX nano_set(points_);
  NanoflannKdTree<NanoPointSetX> tree(
      PointX::Dims,
      nano_set,
      nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
  tree.buildIndex();

  for (auto _ : state) {
    std::vector<std::pair<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      benchmark::DoNotOptimize(
          sum += tree.radiusSearch(
              p.data, squared, results, nanoflann::SearchParams{0, 0, false}));
    }
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtPicoRadius)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 4.0;
  double squared = radius * radius;
  PicoPointSetX pico_set(points_);
  KdTree<PicoPointSetX> tree(pico_set, max_leaf_size);

  for (auto _ : state) {
    std::vector<std::pair<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchRadius(p, squared, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

// Argument 1: Maximum leaf size.
// Argument 2: Search radius (divided by 4.0).
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtNanoRadius)
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

BENCHMARK_REGISTER_F(KdTreeBenchmark, CtPicoRadius)
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
