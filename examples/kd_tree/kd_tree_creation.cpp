#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// The examples in this application demonstrate the different ways in which a
// KdTree can be constructed from a point set. The following is covered:
//
// Value-move idiom:
// A KdTree takes the input by value. This means that the KdTree takes ownership
// of a copy of the input. When a copy is not desired, the point set can either
// be moved into the KdTree or it can take the point set by reference by
// wrapping it in an std::reference_wrapper<>. In the latter case, the KdTree
// will only have shallow ownership of the input. This allows it to be used for
// other purposes as well.
//
// Class template argument deduction:
// The class template argument that defines the space type (the input point set
// type) does not always have to be specified and can be deduced by the
// compiler. In case another class template argument needs to be specified, such
// as the metric type, then the space type may still be deduced using the
// MakeKdTree<> convenience method.

// Although all of the examples use an std::vector<std::array<>> as the input
// for building a KdTree, they will work with any of the inputs supported by
// this library (e.g., Eigen::Matrix<>).
using Space = std::vector<std::array<float, 3>>;

template <typename Tree>
void QueryTree(Tree const& tree) {
  float query[3] = {4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;
}

auto MakePointSet() { return Space{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}}; }

// In this example, the creation of a KdTree results in a copy of the input
// point set. The KdTree has full ownership of the copy.
void BuildKdTreeWithCopy() {
  auto points = MakePointSet();

  pico_tree::KdTree<Space> tree(points, pico_tree::max_leaf_size_t(12));

  QueryTree(tree);
}

// In this example, the point sets are not copied but moved into the KdTrees
// when they are created. Each tree has full ownership of the moved point set.
void BuildKdTreeWithMove() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = MakePointSet();

  pico_tree::KdTree<Space> tree1(std::move(points), max_leaf_size);

  pico_tree::KdTree<Space> tree2(MakePointSet(), max_leaf_size);

  QueryTree(tree1);
  QueryTree(tree2);
}

// In this example, the input is wrapped in an std::reference_wrapper<>. Thus,
// only a reference is copied by a KdTree. Each tree only has shallow ownership
// of the input point set.
void BuildKdTreeWithReference() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = MakePointSet();

  // By reference.
  pico_tree::KdTree<std::reference_wrapper<Space>> tree1(points, max_leaf_size);

  // By const reference.
  pico_tree::KdTree<std::reference_wrapper<Space const>> tree2(
      points, max_leaf_size);

  QueryTree(tree1);
  QueryTree(tree2);
}

// This example shows that the type of the input point set may be deduced by the
// compiler.
void SpaceTypeDeduction() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  auto points = MakePointSet();

  // The type of the first class template argument, the space type, is
  // determined by the compiler.
  pico_tree::KdTree tree1(std::ref(points), max_leaf_size);

  using KdTree1Type = pico_tree::KdTree<std::reference_wrapper<Space>>;

  static_assert(std::is_same_v<decltype(tree1), KdTree1Type>);

  // Using the previous auto deduction method, we still have to specify the
  // space type when we want to change any of the other template arguments, such
  // as the metric type. In this case we can use the MakeKdTree method to make
  // life a bit easier.
  auto tree2 =
      pico_tree::MakeKdTree<pico_tree::LInf>(std::ref(points), max_leaf_size);

  using KdTree2Type =
      pico_tree::KdTree<std::reference_wrapper<Space>, pico_tree::LInf>;

  static_assert(std::is_same_v<decltype(tree2), KdTree2Type>);

  QueryTree(tree1);
  QueryTree(tree2);
}

int main() {
  SpaceTypeDeduction();
  BuildKdTreeWithReference();
  BuildKdTreeWithMove();
  BuildKdTreeWithCopy();
  return 0;
}
