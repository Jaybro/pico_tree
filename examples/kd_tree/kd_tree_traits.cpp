#include <array>
#include <deque>
#include <iostream>
#include <pico_tree/kd_tree.hpp>

// This example shows how to write a traits class for a custom space type (or
// point set type).

namespace pico_tree {

// Provides an interface for std::array<Scalar_, Dim_>.
template <typename Scalar_, std::size_t Dim_>
struct PointTraits<std::array<Scalar_, Dim_>> {
  using PointType = std::array<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  // Returns a pointer to the coordinates of the input point.
  inline static ScalarType const* Coords(PointType const& point) {
    return point.data();
  }

  // Returns the spatial dimension of the input point. Note that the input
  // argument is ignored because the spatial dimension is known at compile time.
  inline static std::size_t constexpr Sdim(PointType const&) { return Dim; }
};

// Provides an interface for an std::deque<std::array>.
template <typename Scalar_, std::size_t Dim_>
struct SpaceTraits<std::deque<std::array<Scalar_, Dim_>>> {
  using SpaceType = std::deque<std::array<Scalar_, Dim_>>;
  using PointType = std::array<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  // Spatial dimension. Set to pico_tree::kDynamicSize when the dimension is
  // only known at run-time.
  static std::size_t constexpr Dim = Dim_;

  // Returns the amount of coordinates of each point.
  inline static std::size_t Sdim(SpaceType const&) { return Dim; }

  // Returns number of points contained by the space.
  inline static std::size_t Npts(SpaceType const& space) {
    return space.size();
  }

  // Returns the idx'th point from the input space.
  template <typename Index_>
  inline static PointType const& PointAt(
      SpaceType const& space, Index_ const idx) {
    return space[static_cast<std::size_t>(idx)];
  }
};

// This version of Traits makes the KdTree take the input by const reference.
template <typename Scalar_, std::size_t Dim_>
struct SpaceTraits<
    std::reference_wrapper<std::deque<std::array<Scalar_, Dim_>>> const>
    : public SpaceTraits<std::deque<std::array<Scalar_, Dim_>>> {
  using SpaceType =
      std::reference_wrapper<std::deque<std::array<Scalar_, Dim_>>> const;
};

}  // namespace pico_tree

template <typename Tree>
void SearchPoint(Tree const& tree) {
  std::array<float, 2> query{4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;
}

int main() {
  int max_leaf_size = 1;
  std::deque<std::array<float, 2>> points{
      {0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}};

  // The KdTree takes the input by value. To prevent a copy, we can either use
  // an std::reference_wrapper or move the points into the tree.

  {
    // A copy can be prevented by using an std::reference_wrapper.
    pico_tree::KdTree<
        std::reference_wrapper<std::deque<std::array<float, 2>>> const>
        tree(points, max_leaf_size);

    SearchPoint(tree);
  }

  {
    // A copy can be prevented by moving the points into the tree.
    pico_tree::KdTree<std::deque<std::array<float, 2>>> tree(
        std::move(points), max_leaf_size);

    SearchPoint(tree);
  }

  return 0;
}
