#include <iostream>
#include <memory>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>

// This example shows how to work with dynamic size arrays. Support is provided
// for working with an array of scalars or an array of points.

void array_of_scalars() {
  std::size_t count = 6;
  constexpr std::size_t dim = 2;

  // Here we create an array of scalars that will be interpreted as a set of
  // points: {{0, 1}, {2, 3}, ...}
  std::unique_ptr<double[]> data = std::make_unique<double[]>(count * dim);
  for (std::size_t i = 0; i < (count * dim); ++i) {
    data[i] = static_cast<double>(i);
  }

  // If dim equals pico_tree::dynamic_size, then space_map will need a 3rd
  // argument: The spatial dimension known at run time.
  pico_tree::space_map<pico_tree::point_map<double, dim>> map(
      data.get(), count);

  pico_tree::kd_tree<pico_tree::space_map<pico_tree::point_map<double, dim>>>
      tree(map, pico_tree::max_leaf_size_t(3));

  // If dim equals pico_tree::dynamic_size, then point_map will need a 2nd
  // argument: The spatial dimension known at run time.
  std::size_t index = 2;
  pico_tree::point_map<double, dim> query(data.get() + index * dim);
  pico_tree::neighbor<int, double> nn;
  tree.search_nn(query, nn);

  // Prints index 2.
  std::cout << "Index closest point: " << nn.index << std::endl;
}

void array_of_points() {
  std::size_t count = 6;
  constexpr std::size_t dim = 2;

  // Here we create an array of points: {{0, 1}, {2, 3}, ...}
  std::unique_ptr<std::array<double, dim>[]> data =
      std::make_unique<std::array<double, dim>[]>(count);
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = 0; j < dim; ++j) {
      data[i][j] = static_cast<double>(i * dim + j);
    }
  }

  pico_tree::space_map<std::array<double, dim>> map(data.get(), count);

  pico_tree::kd_tree<pico_tree::space_map<std::array<double, dim>>> tree(
      map, pico_tree::max_leaf_size_t(3));

  std::size_t index = 1;
  std::array<double, dim> const& query = data[index];
  pico_tree::neighbor<int, double> nn;
  tree.search_nn(query, nn);

  // Prints index 1.
  std::cout << "Index closest point: " << nn.index << std::endl;
}

int main() {
  array_of_scalars();
  array_of_points();
  return 0;
}
