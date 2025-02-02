#include <iostream>
#include <memory>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>

// This example shows how to work with dynamic size arrays. Support is provided
// for working with an array of scalars or an array of points.

void ArrayOfScalars() {
  std::size_t count = 6;
  constexpr std::size_t Dim = 2;

  // Here we create an array of scalars that will be interpreted as a set of
  // points: {{0, 1}, {2, 3}, ...}
  std::unique_ptr<double[]> data = std::make_unique<double[]>(count * Dim);
  for (std::size_t i = 0; i < (count * Dim); ++i) {
    data[i] = static_cast<double>(i);
  }

  // If Dim equals pico_tree::kDynamicSize, then SpaceMap will need a 3rd
  // argument: The spatial dimension known at run time.
  pico_tree::SpaceMap<pico_tree::PointMap<double, Dim>> map(data.get(), count);

  pico_tree::KdTree<pico_tree::SpaceMap<pico_tree::PointMap<double, Dim>>> tree(
      map, pico_tree::max_leaf_size_t(3));

  // If Dim equals pico_tree::kDynamicSize, then PointMap will need a 2nd
  // argument: The spatial dimension known at run time.
  std::size_t index = 2;
  pico_tree::PointMap<double, Dim> query(data.get() + index * Dim);
  pico_tree::Neighbor<int, double> nn;
  tree.SearchNn(query, nn);

  // Prints index 2.
  std::cout << "Index closest point: " << nn.index << std::endl;
}

void ArrayOfPoints() {
  std::size_t count = 6;
  constexpr std::size_t Dim = 2;

  // Here we create an array of points: {{0, 1}, {2, 3}, ...}
  std::unique_ptr<std::array<double, Dim>[]> data =
      std::make_unique<std::array<double, Dim>[]>(count);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < Dim; ++j) {
      data[i][j] = static_cast<double>(i * Dim + j);
    }
  }

  pico_tree::SpaceMap<std::array<double, Dim>> map(data.get(), count);

  pico_tree::KdTree<pico_tree::SpaceMap<std::array<double, Dim>>> tree(
      map, pico_tree::max_leaf_size_t(3));

  std::size_t index = 1;
  std::array<double, Dim> const& query = data[index];
  pico_tree::Neighbor<int, double> nn;
  tree.SearchNn(query, nn);

  // Prints index 1.
  std::cout << "Index closest point: " << nn.index << std::endl;
}

int main() {
  ArrayOfScalars();
  ArrayOfPoints();
  return 0;
}
