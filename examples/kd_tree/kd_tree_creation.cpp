#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// The examples in this application demonstrate the different ways in which a
// kd_tree can be constructed from a point set. The following is covered:
//
// Value-move idiom:
// A kd_tree takes the input by value. This means that the kd_tree takes
// ownership of a copy of the input. When a copy is not desired, the point set
// can either be moved into the kd_tree or it can take the point set by
// reference by wrapping it in an std::reference_wrapper<>. In the latter case,
// the kd_tree will only have shallow ownership of the input. This allows it to
// be used for other purposes as well.
//
// Class template argument deduction:
// The class template argument that defines the space type (the input point set
// type) does not always have to be specified and can be deduced by the
// compiler. In case another class template argument needs to be specified, such
// as the metric type, then the space type may still be deduced using the
// make_kd_tree<> convenience method.

// Although all of the examples use an std::vector<std::array<>> as the input
// for building a kd_tree, they will work with any of the inputs supported by
// this library (e.g., Eigen::Matrix<>).
using space = std::vector<std::array<float, 3>>;

template <typename Tree_>
void query_tree(Tree_ const& tree) {
  float query[3] = {4.0f, 4.0f, 4.0f};
  pico_tree::neighbor<int, float> nn;
  tree.search_nn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;
}

auto make_point_set() { return space{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}}; }

// In this example, the creation of a kd_tree results in a copy of the input
// point set. The kd_tree has full ownership of the copy.
void build_kd_tree_with_copy() {
  auto points = make_point_set();

  pico_tree::kd_tree<space> tree(points, pico_tree::max_leaf_size_t(12));

  query_tree(tree);
}

// In this example, the point sets are not copied but moved into the KdTrees
// when they are created. Each tree has full ownership of the moved point set.
void build_kd_tree_with_move() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = make_point_set();

  pico_tree::kd_tree<space> tree1(std::move(points), max_leaf_size);

  pico_tree::kd_tree<space> tree2(make_point_set(), max_leaf_size);

  query_tree(tree1);

  query_tree(tree2);
}

// In this example, the input is wrapped in an std::reference_wrapper<>. Thus,
// only a reference is copied by a kd_tree. Each tree only has shallow ownership
// of the input point set.
void build_kd_tree_with_reference() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = make_point_set();

  // By reference.
  pico_tree::kd_tree<std::reference_wrapper<space>> tree1(
      points, max_leaf_size);

  // By const reference.
  pico_tree::kd_tree<std::reference_wrapper<space const>> tree2(
      points, max_leaf_size);

  query_tree(tree1);

  query_tree(tree2);
}

// This example shows that the type of the input point set may be deduced by the
// compiler.
void space_type_deduction() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = make_point_set();

  // The type of the first class template argument, the space type, is
  // determined by the compiler.
  pico_tree::kd_tree tree1(std::ref(points), max_leaf_size);

  using kd_tree1_type = pico_tree::kd_tree<std::reference_wrapper<space>>;

  static_assert(std::is_same_v<decltype(tree1), kd_tree1_type>);

  // Using the previous auto deduction method, we still have to specify the
  // space type when we want to change any of the other template arguments, such
  // as the metric type. In this case we can use the make_kd_tree method to make
  // life a bit easier.
  auto tree2 = pico_tree::make_kd_tree<pico_tree::metric_linf>(
      std::ref(points), max_leaf_size);

  using kd_tree2_type =
      pico_tree::kd_tree<std::reference_wrapper<space>, pico_tree::metric_linf>;

  static_assert(std::is_same_v<decltype(tree2), kd_tree2_type>);

  query_tree(tree1);

  query_tree(tree2);
}

int main() {
  space_type_deduction();
  build_kd_tree_with_reference();
  build_kd_tree_with_move();
  build_kd_tree_with_copy();
  return 0;
}
