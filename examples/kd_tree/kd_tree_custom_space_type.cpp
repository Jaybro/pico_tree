#include <deque>
#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>

// This example shows how to write a traits class for a custom space type (point
// set type).

namespace pico_tree {

// Provides an interface for an std::deque<std::array>.
template <typename Scalar_, std::size_t Dim_>
struct SpaceTraits<std::deque<std::array<Scalar_, Dim_>>> {
  using SpaceType = std::deque<std::array<Scalar_, Dim_>>;
  using PointType = std::array<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  // Spatial dimension. Set to pico_tree::kDynamicSize when the dimension is
  // only known at run-time.
  static std::size_t constexpr Dim = Dim_;

  // Returns a point from the input space at the specified index.
  template <typename Index_>
  inline static PointType const& PointAt(
      SpaceType const& space, Index_ const index) {
    return space[static_cast<std::size_t>(index)];
  }

  // Returns number of points contained by the space.
  inline static std::size_t size(SpaceType const& space) {
    return space.size();
  }

  // Returns the number of coordinates or spatial dimension of each point.
  inline static constexpr std::size_t sdim(SpaceType const&) { return Dim; }
};

}  // namespace pico_tree

int main() {
  std::size_t max_leaf_size = 1;
  std::deque<std::array<float, 2>> points{
      {0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}};

  pico_tree::KdTree<std::reference_wrapper<std::deque<std::array<float, 2>>>>
      tree(points, max_leaf_size);

  std::array<float, 2> query{4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
