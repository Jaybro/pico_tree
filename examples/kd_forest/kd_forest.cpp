#include <iostream>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>
#include <pico_understory/kd_forest.hpp>

#include "mnist.hpp"
#include "sift.hpp"

template <typename Vector_, typename Scalar_>
std::size_t run_kd_forest(
    std::vector<Vector_> const& train,
    std::vector<Vector_> const& test,
    std::vector<pico_tree::neighbor<int, Scalar_>> const& nns,
    std::size_t forest_size,
    std::size_t forest_max_leaf_size,
    std::size_t forest_max_leaves_visited) {
  using space = std::reference_wrapper<std::vector<Vector_> const>;

  auto rkd_tree = [&train, &forest_max_leaf_size, &forest_size]() {
    pico_tree::scoped_timer t0("kd_forest build");
    return pico_tree::kd_forest<space>(
        train, forest_max_leaf_size, forest_size);
  }();

  pico_tree::scoped_timer t1("kd_forest query");
  pico_tree::neighbor<int, Scalar_> nn;

  std::size_t equal = 0;
  for (std::size_t i = 0; i < nns.size(); ++i) {
    rkd_tree.search_nn(test[i], forest_max_leaves_visited, nn);

    if (nns[i].index == nn.index) {
      ++equal;
    }
  }

  return equal;
}

template <typename Vector_, typename Scalar_>
void run_kd_tree(
    std::vector<Vector_> const& train,
    std::vector<Vector_> const& test,
    std::string const& fn_nns_gt,
    pico_tree::max_leaf_size_t tree_max_leaf_size,
    std::vector<pico_tree::neighbor<int, Scalar_>>& nns) {
  using space = std::reference_wrapper<std::vector<Vector_> const>;

  nns.resize(test.size());

  if (!std::filesystem::exists(fn_nns_gt)) {
    std::cout << "Creating " << fn_nns_gt
              << " using the kd_tree. Be *very* patient." << std::endl;

    auto kd_tree = [&train, &tree_max_leaf_size]() {
      pico_tree::scoped_timer t0("kd_tree build");
      return pico_tree::kd_tree<space>(train, tree_max_leaf_size);
    }();

    {
      pico_tree::scoped_timer t1("kd_tree query");
      for (std::size_t i = 0; i < nns.size(); ++i) {
        kd_tree.search_nn(test[i], nns[i]);
      }
    }

    pico_tree::write_bin(fn_nns_gt, nns);
  } else {
    pico_tree::read_bin(fn_nns_gt, nns);
    std::cout << "kd_tree not created. Read " << fn_nns_gt << " instead."
              << std::endl;
  }
}

// A kd_forest takes roughly forest_size times longer to build compared to
// building a kd_tree. However, the kd_forest is usually a lot faster with
// queries in high dimensions with the added trade-off that the exact nearest
// neighbor may not be found.
template <typename Dataset_>
void run_dataset(
    std::size_t tree_max_leaf_size,
    std::size_t forest_size,
    std::size_t forest_max_leaf_size,
    std::size_t forest_max_leaves_visited) {
  using Point = typename Dataset_::point_type;
  using scalar_type = typename Point::value_type;

  auto train = Dataset_::read_train();
  auto test = Dataset_::read_test();
  std::vector<pico_tree::neighbor<int, scalar_type>> nns;
  std::string fn_nns_gt = Dataset_::dataset_name + "_nns_gt.bin";

  run_kd_tree(train, test, fn_nns_gt, tree_max_leaf_size, nns);

  std::size_t equal = run_kd_forest(
      train,
      test,
      nns,
      forest_size,
      forest_max_leaf_size,
      forest_max_leaves_visited);

  std::cout << std::setprecision(10);
  std::cout << "Precision: "
            << (static_cast<float>(equal) / static_cast<float>(nns.size()))
            << std::endl;
}

int main() {
  // forest_max_leaf_size = 16
  // forest_max_leaves_visited = 16
  //    forest_size 8: a precision of around 0.915.
  //    forest_size 16: a precision of around 0.976.
  run_dataset<mnist>(16, 8, 16, 16);
  // forest_max_leaf_size = 32
  // forest_max_leaves_visited = 64
  //    forest_size 8: a precision of around 0.884.
  //    forest_size 16: a precision of around 0.940.
  //    forest_size 128: out of memory :'(
  run_dataset<sift>(16, 8, 32, 64);
  return 0;
}
