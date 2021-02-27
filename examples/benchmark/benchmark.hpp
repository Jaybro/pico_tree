#include <benchmark/benchmark.h>

#include "format_bin.hpp"

namespace pico_tree {

class Benchmark : public benchmark::Fixture {
 protected:
  using Index = int;
  using Scalar = double;
  using PointX = Point3d;

 public:
  Benchmark() {
    // Here you may need to be patient. Depending on the size of the compiled
    // binary.
    pico_tree::ReadBin("./scans.bin", &points_);
  }

 protected:
  std::vector<PointX> points_;
};

}  // namespace pico_tree
