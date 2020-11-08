#include <benchmark/benchmark.h>

#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

#include "format_bin.hpp"
#include "nano_adaptor.hpp"

template <int Dims, typename PicoAdaptor>
using MetricL2 = pico_tree::MetricL2<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    Dims,
    PicoAdaptor>;

template <int Dims, typename PicoAdaptor>
using SplitterLongestMedian = pico_tree::SplitterLongestMedian<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeCtSldMid = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dim,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeCtLngMed = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dim,
    PicoAdaptor,
    MetricL2<PicoAdaptor::Dim, PicoAdaptor>,
    SplitterLongestMedian<PicoAdaptor::Dim, PicoAdaptor>>;

template <typename PicoAdaptor>
using PicoKdTreeRtSldMid = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    pico_tree::kRuntimeDims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using PicoKdTreeRtLngMed = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    pico_tree::kRuntimeDims,
    PicoAdaptor,
    MetricL2<pico_tree::kRuntimeDims, PicoAdaptor>,
    SplitterLongestMedian<pico_tree::kRuntimeDims, PicoAdaptor>>;

template <typename NanoAdaptor>
using NanoKdTreeCt = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<typename NanoAdaptor::Scalar, NanoAdaptor>,
    NanoAdaptor,
    NanoAdaptor::Dims,
    typename NanoAdaptor::Index>;

template <typename NanoAdaptor>
using NanoKdTreeRt = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<typename NanoAdaptor::Scalar, NanoAdaptor>,
    NanoAdaptor,
    -1,
    typename NanoAdaptor::Index>;

class KdTreeBenchmark : public benchmark::Fixture {
 protected:
  using Index = int;
  using Scalar = double;
  using PointX = Point3d;
  using NanoAdaptorX = NanoAdaptor<Index, PointX>;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

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
  NanoAdaptorX adaptor(points_);
  for (auto _ : state) {
    NanoKdTreeCt<NanoAdaptorX> tree(
        PointX::Dim,
        adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtSldMidPicoBuildTree)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtLngMedPicoBuildTree)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeCtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, RtNanoBuildTree)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  NanoAdaptorX adaptor(points_);
  for (auto _ : state) {
    NanoKdTreeRt<NanoAdaptorX> tree(
        PointX::Dim,
        adaptor,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_size));
    tree.buildIndex();
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, RtSldMidPicoBuildTree)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeRtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, RtLngMedPicoBuildTree)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  PicoAdaptorX adaptor(points_);
  for (auto _ : state) {
    PicoKdTreeRtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

// Argument 1: Maximum leaf size.
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtNanoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtSldMidPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtLngMedPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

BENCHMARK_REGISTER_F(KdTreeBenchmark, RtNanoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, RtSldMidPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);
BENCHMARK_REGISTER_F(KdTreeBenchmark, RtLngMedPicoBuildTree)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->DenseRange(6, 14, 2);

// ****************************************************************************
// Knn
// ****************************************************************************

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtNanoKnn)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  NanoAdaptorX adaptor(points_);
  NanoKdTreeCt<NanoAdaptorX> tree(
      PointX::Dim,
      adaptor,
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

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtSldMidPicoKnn)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);

  for (auto _ : state) {
    std::vector<std::pair<Index, Scalar>> results;
    std::size_t sum = 0;
    for (auto const& p : points_) {
      tree.SearchKnn(p, knn_count, &results);
      benchmark::DoNotOptimize(sum += results.size());
    }
  }
}

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtLngMedPicoKnn)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  int knn_count = state.range(1);
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtLngMed<PicoAdaptorX> tree(adaptor, max_leaf_size);

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

BENCHMARK_REGISTER_F(KdTreeBenchmark, CtSldMidPicoKnn)
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

// A single neighbor is faster, more neighbors slower.
BENCHMARK_REGISTER_F(KdTreeBenchmark, CtLngMedPicoKnn)
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
    ->Args({14, 4});

// ****************************************************************************
// Radius
// ****************************************************************************

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtNanoRadius)(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 4.0;
  double squared = radius * radius;
  NanoAdaptorX adaptor(points_);
  NanoKdTreeCt<NanoAdaptorX> tree(
      PointX::Dim,
      adaptor,
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

BENCHMARK_DEFINE_F(KdTreeBenchmark, CtSldMidPicoRadius)
(benchmark::State& state) {
  int max_leaf_size = state.range(0);
  double radius = static_cast<double>(state.range(1)) / 4.0;
  double squared = radius * radius;
  PicoAdaptorX adaptor(points_);
  PicoKdTreeCtSldMid<PicoAdaptorX> tree(adaptor, max_leaf_size);

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

BENCHMARK_REGISTER_F(KdTreeBenchmark, CtSldMidPicoRadius)
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
