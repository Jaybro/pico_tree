#include <iostream>
// Provides support for fixed size arrays and std::array.
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
// Provides support for std::vector.
#include <pico_tree/vector_traits.hpp>

// This example shows how to create and query a kd_tree. An std::vector can be
// used for storing points and a point can be either an array or an std::array.

int main() {
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::vector<std::array<float, 3>> points{
      {0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  // The kd_tree takes the input by value. To prevent a copy, we can either move
  // the point set into the tree or the point set can be taken by reference by
  // wrapping it in an std::reference_wrapper. Below we take the input by
  // reference:
  pico_tree::kd_tree tree(std::ref(points), max_leaf_size);

  float query[3] = {4.0f, 4.0f, 4.0f};
  pico_tree::neighbor<int, float> nn;
  tree.search_nn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
