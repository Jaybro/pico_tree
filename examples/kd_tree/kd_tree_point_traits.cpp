#include <iostream>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/std_traits.hpp>

// This example shows how to write a small traits class for a custom point type
// when the points are stored in an std::vector. Support for using an
// std::vector is already provided by PicoTree via pico_tree::StdTraits.

struct PointXYZ {
  float x;
  float y;
  float z;
};

// A specialization of StdPointTraits must be defined within the pico_tree
// namespace and provide all the details of this example.
namespace pico_tree {

template <>
struct StdPointTraits<PointXYZ> {
  static_assert(sizeof(PointXYZ) == sizeof(float) * 3, "");
  using ScalarType = float;
  static constexpr int Dim = 3;

  // Returns a pointer to the coordinates of the input point.
  inline static ScalarType const* Coords(PointXYZ const& point) {
    return &point.x;
  }

  // Returns the spatial dimension of the input point. Note that the input
  // argument is ignored because the spatial dimension is known at compile time.
  inline static int Sdim(PointXYZ const&) { return Dim; }
};

}  // namespace pico_tree

int main() {
  int max_leaf_size = 12;
  std::vector<PointXYZ> points{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  // Using an std::reference_wrapper prevents the KdTree from making a copy of
  // the input. Not using it allows the point set to be owned by the tree, in
  // which case a copy can be prevented by moving the points into the tree.
  pico_tree::KdTree<
      pico_tree::StdTraits<std::reference_wrapper<std::vector<PointXYZ>>>>
      tree(points, max_leaf_size);

  PointXYZ query{4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, &nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
