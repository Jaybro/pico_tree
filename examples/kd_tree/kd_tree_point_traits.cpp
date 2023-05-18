#include <iostream>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/std_traits.hpp>

// This example shows how to write a traits class for a custom point type that
// is stored in an std::vector.

// PointXYZ is based on the Point Cloud Library (PCL). Within PCL, PointXYZ is
// actually a union between a float data[4] and a struct { float x, y, z; }, but
// the traits we define here would still work.
struct PointXYZ {
  float data[3];
};

// A specialization of PointTraits must be defined within the pico_tree
// namespace and provide all the details of this example.
namespace pico_tree {

template <>
struct PointTraits<PointXYZ> {
  using ScalarType = float;
  static std::size_t constexpr Dim = 3;

  // Returns a pointer to the coordinates of the input point.
  inline static ScalarType const* Coords(PointXYZ const& point) {
    return point.data;
  }

  // Returns the spatial dimension of the input point. Note that the input
  // argument is ignored because the spatial dimension is known at compile time.
  inline static std::size_t Sdim(PointXYZ const&) { return Dim; }
};

}  // namespace pico_tree

int main() {
  int max_leaf_size = 12;
  std::vector<PointXYZ> points{{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}};

  // Using an std::reference_wrapper prevents the KdTree from making a copy of
  // the input. The KdTree can take ownership of the pointset if we omit the
  // std::reference_wrapper and move the pointset inside the tree.
  pico_tree::KdTree<
      pico_tree::StdTraits<std::reference_wrapper<std::vector<PointXYZ>>>>
      tree(points, max_leaf_size);

  PointXYZ query{4.0f, 4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, &nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
