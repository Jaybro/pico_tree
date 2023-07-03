#include <iostream>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// This example shows how to write a traits class for a custom point type.

// PointXYZ is the custom point type for which we write a simple traits class.
struct PointXYZ {
  float data[3];
};

// A specialization of PointTraits must be defined within the pico_tree
// namespace and provide all the details of this example.
namespace pico_tree {

template <>
struct PointTraits<PointXYZ> {
  using PointType = PointXYZ;
  using ScalarType = float;
  // Spatial dimension. Set to pico_tree::kDynamicSize when the dimension is
  // only known at run-time.
  static std::size_t constexpr Dim = 3;

  // Returns a pointer to the coordinates of the input point.
  inline static float const* data(PointXYZ const& point) { return point.data; }

  // Returns the number of coordinates or spatial dimension of each point.
  inline static constexpr std::size_t size(PointXYZ const&) { return Dim; }
};

}  // namespace pico_tree

int main() {
  std::size_t max_leaf_size = 12;
  std::vector<PointXYZ> points{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  pico_tree::KdTree<std::reference_wrapper<std::vector<PointXYZ>>> tree(
      points, max_leaf_size);

  PointXYZ query{4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
