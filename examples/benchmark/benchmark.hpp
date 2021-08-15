#include <benchmark/benchmark.h>

#include <pico_toolshed/point.hpp>

#include "format_bin.hpp"

// It seems that there is a "threshold" to the amount of functions being
// benchmarked. Having "too many" of them makes them become slow(er). This
// appears to be due to code alignment. See similar issue and video referencing
// that issue:
//
// * https://github.com/google/benchmark/issues/461
// * https://www.youtube.com/watch?v=10MQW-aJU3g&t=197s
//
// Having split up many of the benchmarks into different executables solved it
// for a while, but the problem came back for bm_pico_kd_tree. For now,
// benchmarks are just enabled 1-by-1.

namespace pico_tree {

class Benchmark : public benchmark::Fixture {
 protected:
  using Index = int;
  using Scalar = float;
  using PointX = Point3f;

 public:
  Benchmark() {
    // Here you may need to be patient depending on the size of the binaries.
    // Loaded for each benchmark.
    pico_tree::ReadBin("./scans0.bin", &points_tree_);
    pico_tree::ReadBin("./scans1.bin", &points_test_);
  }

 protected:
  std::vector<PointX> points_tree_;
  std::vector<PointX> points_test_;
};

}  // namespace pico_tree
