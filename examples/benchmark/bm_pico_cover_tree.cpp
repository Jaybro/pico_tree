#include <pico_adaptor.hpp>
#include <pico_tree/cover_tree.hpp>

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

BENCHMARK_MAIN();
