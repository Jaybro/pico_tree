#include <iostream>
#include <memory>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>

// This example shows how to work with dynamic size arrays. Support is provided
// for working with an array of scalars or an array of points.

void ArrayOfScalars() {
  std::size_t count = 10;
  constexpr std::size_t Dim = 2;
  std::size_t max_leaf_size = 3;

  // Dummy array of scalars.
  std::unique_ptr<double[]> data = std::make_unique<double[]>(count);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = static_cast<double>(i);
  }

  // If Dim equals pico_tree::kDynamicSize, SpaceMap needs a 3rd argument: The
  // spatial dimension known at run time.
  pico_tree::SpaceMap<pico_tree::PointMap<double, Dim>> map(
      data.get(), count / Dim);

  pico_tree::KdTree<pico_tree::SpaceMap<pico_tree::PointMap<double, Dim>>> tree(
      map, max_leaf_size);

  // If Dim equals pico_tree::kDynamicSize, PointMap needs a 2nd argument: The
  // spatial dimension known at run time.
  pico_tree::PointMap<double, Dim> query(data.get() + 4);
  pico_tree::Neighbor<int, double> nn;
  tree.SearchNn(query, nn);
  // Prints index 2.
  std::cout << "Index closest point: " << nn.index << std::endl;
}

void ArrayOfPoints() {
  std::size_t count = 3;
  constexpr std::size_t Dim = 2;
  std::size_t max_leaf_size = 3;

  // Dummy array of points.
  std::unique_ptr<std::array<double, Dim>[]> data =
      std::make_unique<std::array<double, Dim>[]>(count);
  for (std::size_t i = 0; i < count; ++i) {
    data[i] = {
        static_cast<double>(i * Dim + 0), static_cast<double>(i * Dim + 1)};
  }

  pico_tree::SpaceMap<std::array<double, Dim>> map(data.get(), count);

  pico_tree::KdTree<pico_tree::SpaceMap<std::array<double, Dim>>> tree(
      map, max_leaf_size);

  std::array<double, Dim> const& query = *(data.get() + 1);
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
