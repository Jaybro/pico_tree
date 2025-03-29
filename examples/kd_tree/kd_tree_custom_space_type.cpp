#include <deque>
#include <iostream>
#include <pico_tree/array_traits.hpp>
#include <pico_tree/kd_tree.hpp>

// This example shows how to write a traits class for a custom space type (point
// set type).

namespace pico_tree {

// Provides an interface for an std::deque<std::array>.
template <typename Scalar_, std::size_t Dim_>
struct space_traits<std::deque<std::array<Scalar_, Dim_>>> {
  using space_type = std::deque<std::array<Scalar_, Dim_>>;
  using point_type = std::array<Scalar_, Dim_>;
  using scalar_type = Scalar_;
  // Spatial dimension. Set to pico_tree::dynamic_extent when the dimension is
  // only known at run-time.
  static constexpr pico_tree::size_t dim = Dim_;

  // Returns a point from the input space at the specified index.
  template <typename Index_>
  inline static point_type const& point_at(
      space_type const& space, Index_ const index) {
    return space[static_cast<std::size_t>(index)];
  }

  // Returns number of points contained by the space.
  inline static pico_tree::size_t size(space_type const& space) {
    return space.size();
  }

  // Returns the number of coordinates or spatial dimension of each point.
  inline static constexpr pico_tree::size_t sdim(space_type const&) {
    return dim;
  }
};

}  // namespace pico_tree

int main() {
  std::deque<std::array<float, 2>> points{
      {0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}};

  pico_tree::kd_tree<std::reference_wrapper<std::deque<std::array<float, 2>>>>
      tree(points, pico_tree::max_leaf_size_t(1));

  std::array<float, 2> query{4.0f, 4.0f};
  pico_tree::neighbor<int, float> nn;
  tree.search_nn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
