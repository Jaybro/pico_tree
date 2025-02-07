#include <iostream>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// This example shows how to write a traits class for a custom point type.

// PointXYZ is the custom point type for which we write a simple traits class.
struct PointXYZ {
  float data[3];
};

// A specialization of point_traits must be defined within the pico_tree
// namespace and provide all the details of this example.
namespace pico_tree {

template <>
struct point_traits<PointXYZ> {
  using point_type = PointXYZ;
  using scalar_type = float;
  // Spatial dimension. Set to pico_tree::dynamic_size when the dimension is
  // only known at run-time.
  static constexpr pico_tree::size_t dim = 3;

  // Returns a pointer to the coordinates of the input point.
  inline static float const* data(PointXYZ const& point) { return point.data; }

  // Returns the number of coordinates or spatial dimension of each point.
  inline static constexpr pico_tree::size_t size(PointXYZ const&) {
    return dim;
  }
};

}  // namespace pico_tree

int main() {
  std::vector<PointXYZ> points{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  pico_tree::kd_tree<std::reference_wrapper<std::vector<PointXYZ>>> tree(
      points, pico_tree::max_leaf_size_t(12));

  PointXYZ query{4.0f, 4.0f, 4.0f};
  pico_tree::neighbor<int, float> nn;
  tree.search_nn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
