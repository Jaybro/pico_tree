#include <iostream>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/kd_forest.hpp>

#include "mnist.hpp"
#include "sift.hpp"

// A KdForest takes roughly forest_size times longer to build compared to
// building a KdTree. However, the KdForest is usually a lot faster with queries
// in high dimensions with the added trade-off that the exact nearest neighbor
// may not be found.
template <typename Dataset>
void RunDataset(
    std::size_t tree_max_leaf_size,
    std::size_t forest_size,
    std::size_t forest_max_leaf_size,
    std::size_t forest_max_leaves_visited) {
  using Point = typename Dataset::PointType;
  using Space = std::reference_wrapper<std::vector<Point>>;
  using Scalar = typename Point::value_type;

  auto train = Dataset::ReadTrain();
  auto test = Dataset::ReadTest();
  std::size_t count = test.size();
  std::vector<pico_tree::Neighbor<int, Scalar>> nns(count);
  std::string fn_nns_gt = Dataset::kDatasetName + "_nns_gt.bin";

  if (!std::filesystem::exists(fn_nns_gt)) {
    std::cout << "Creating " << fn_nns_gt
              << " using the KdTree. Be *very* patient." << std::endl;

    auto kd_tree = [&train, &tree_max_leaf_size]() {
      ScopedTimer t0("kd_tree build");
      return pico_tree::KdTree<Space>(train, tree_max_leaf_size);
    }();

    {
      ScopedTimer t1("kd_tree query");
      for (std::size_t i = 0; i < nns.size(); ++i) {
        kd_tree.SearchNn(test[i], nns[i]);
      }
    }

    pico_tree::WriteBin(fn_nns_gt, nns);
  } else {
    pico_tree::ReadBin(fn_nns_gt, nns);
    std::cout << "KdTree not created. Read " << fn_nns_gt << " instead."
              << std::endl;
  }

  std::size_t equal = 0;
  {
    auto rkd_tree = [&train, &forest_max_leaf_size, &forest_size]() {
      ScopedTimer t0("kd_forest build");
      return pico_tree::KdForest<Space>(
          train, forest_max_leaf_size, forest_size);
    }();

    ScopedTimer t1("kd_forest query");
    pico_tree::Neighbor<int, Scalar> nn;
    for (std::size_t i = 0; i < nns.size(); ++i) {
      rkd_tree.SearchNn(test[i], forest_max_leaves_visited, nn);

      if (nns[i].index == nn.index) {
        ++equal;
      }
    }
  }

  std::cout << "Precision: "
            << (static_cast<float>(equal) / static_cast<float>(count))
            << std::endl;
}

int main() {
  // forest_max_leaf_size = 16
  // forest_max_leaves_visited = 16
  //    forest_size 8: a precision of around 0.915.
  //    forest_size 16: a precision of around 0.976.
  RunDataset<Mnist>(16, 8, 16, 16);
  // forest_max_leaf_size = 32
  // forest_max_leaves_visited = 64
  //    forest_size 8: a precision of around 0.884.
  //    forest_size 16: a precision of around 0.940.
  //    forest_size 128: out of memory :'(
  RunDataset<Sift>(16, 8, 32, 64);
  return 0;
}
