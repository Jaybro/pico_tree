#include <filesystem>
#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// This example shows how to save and load a KdTree to and from a file. Saving
// and loading the KdTree does not include saving and loading the point set.

// A KdTree is stored in a binary format that is architecture dependent. As
// such, saving and loading a file will fail when the file is exchanged between
// machines with different endianness, etc.

int main() {
  std::vector<std::array<float, 3>> points{
      {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  using KdTree = pico_tree::KdTree<
      std::reference_wrapper<std::vector<std::array<float, 3>>>>;

  std::string filename = "tree.bin";

  // Save to file.
  std::size_t max_leaf_size = 12;
  KdTree::Save(KdTree(points, max_leaf_size), filename);

  // Load from file.
  auto tree = KdTree::Load(points, filename);
  std::filesystem::remove(filename);

  float query[3] = {4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
