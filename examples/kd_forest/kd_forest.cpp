#include <iostream>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/kd_forest.hpp>

#include "mnist.hpp"
#include "sift.hpp"

template <typename Dataset>
void RunDataset(
    std::size_t max_leaf_size_exact,
    std::size_t max_leaf_size_apprx,
    std::size_t forest_size) {
  using Point = typename Dataset::PointType;
  using Space = std::reference_wrapper<std::vector<Point>>;
  using Scalar = typename Point::value_type;

  auto train = Dataset::ReadTrain();
  auto test = Dataset::ReadTest();
  std::size_t count = test.size();
  std::vector<pico_tree::Neighbor<int, Scalar>> nns(count);
  std::string fn_nns_gt = Dataset::kDatasetName + "_nns_gt.bin";

  if (!std::filesystem::exists(fn_nns_gt)) {
    auto kd_tree = [&train, &max_leaf_size_exact]() {
      ScopedTimer t0("kd_tree build");
      return pico_tree::KdTree<Space>(train, max_leaf_size_exact);
    }();

    {
      ScopedTimer t1("kd_tree query");
      for (std::size_t i = 0; i < nns.size(); ++i) {
        kd_tree.SearchNn(test[i], nns[i]);
      }
    }

    std::cout << "Writing " << fn_nns_gt << "." << std::endl;
    pico_tree::WriteBin(fn_nns_gt, nns);
  } else {
    std::cout << "Reading " << fn_nns_gt << "." << std::endl;
    pico_tree::ReadBin(fn_nns_gt, nns);
  }

  std::size_t equal = 0;

  // Building the KdForest takes roughly forest_size times longer compared to
  // building the regular KdTree. However, it is usually a lot faster.
  {
    auto rkd_tree = [&train, &max_leaf_size_apprx, &forest_size]() {
      ScopedTimer t0("kd_forest build");
      return pico_tree::KdForest<Space>(
          train, max_leaf_size_apprx, forest_size);
    }();

    ScopedTimer t1("kd_forest query");
    pico_tree::Neighbor<int, Scalar> nn;
    for (std::size_t i = 0; i < nns.size(); ++i) {
      rkd_tree.SearchNn(test[i], nn);

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
  // max_leaf_size_apprx = 128:
  // forest_size 8: a precision of around 79%.
  // forest_size 16: a precision of around 93%.
  // forest_size 32: a precision of around 98%.
  RunDataset<Mnist>(16, 128, 8);
  // max_leaf_size_apprx = 1024:
  // forest_size 8: a precision of around 58%.
  // forest_size 16: a precision of around 68%.
  // forest_size 32: a precision of around 80%.
  // forest_size 64: a precision of around 87%.
  // forest_size 128: out of memory :'(
  RunDataset<Sift>(16, 1024, 8);
  return 0;
}
