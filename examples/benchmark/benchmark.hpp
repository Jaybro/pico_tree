#include <benchmark/benchmark.h>

#include <pico_toolshed/point.hpp>

#include "format_bin.hpp"

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
