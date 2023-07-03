#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// This application demonstrates how a KdTree can take its input point set.
// Although all of the examples use an std::vector<> as the input for building a
// KdTree, they will work with any of the inputs supported by this library
// (e.g., Eigen::Matrix<>).

template <typename Tree>
void QueryTree(Tree const& tree) {
  float query[3] = {4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;
}

auto MakePointSet() {
  std::vector<std::array<float, 3>> points{
      {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  return points;
}

// The KdTree takes the input by value. In this example, creating a KdTree
// results in a copy of the input point set.
void BuildKdTreeWithCopy() {
  int max_leaf_size = 12;
  auto points = MakePointSet();

  pico_tree::KdTree<std::vector<std::array<float, 3>>> tree(
      points, max_leaf_size);

  QueryTree(tree);
}

// The KdTree takes the input by value. In this example, the point sets are not
// copied but moved into the KdTree. This prevents a copy.
void BuildKdTreeWithMove() {
  int max_leaf_size = 12;
  auto points = MakePointSet();

  pico_tree::KdTree<std::vector<std::array<float, 3>>> tree1(
      std::move(points), max_leaf_size);

  pico_tree::KdTree<std::vector<std::array<float, 3>>> tree2(
      MakePointSet(), max_leaf_size);

  QueryTree(tree1);
  QueryTree(tree2);
}

// The KdTree takes the input by value. In this example, the input is taken by
// reference. This prevents a copy.
void BuildKdTreeWithReference() {
  int max_leaf_size = 12;
  auto points = MakePointSet();

  // By reference.
  pico_tree::KdTree<std::reference_wrapper<std::vector<std::array<float, 3>>>>
      tree1(points, max_leaf_size);

  // By const reference.
  pico_tree::KdTree<
      std::reference_wrapper<std::vector<std::array<float, 3>> const>>
      tree2(points, max_leaf_size);

  QueryTree(tree1);
  QueryTree(tree2);
}

int main() {
  BuildKdTreeWithReference();
  BuildKdTreeWithMove();
  BuildKdTreeWithCopy();
  return 0;
}
