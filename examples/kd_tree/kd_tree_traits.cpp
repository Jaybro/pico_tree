#include <array>
#include <deque>
#include <iostream>
#include <pico_tree/kd_tree.hpp>

// This example shows how to write a traits class for a custom space type (or
// point set type).

// Traits provides an interface for an std::deque<std::array>.
template <typename Scalar_, std::size_t Dim_>
struct Traits {
  using SpaceType = std::deque<std::array<Scalar_, Dim_>>;
  using PointType = std::array<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  // The index type of point coordinates stored by the tree.
  using IndexType = int;
  // Spatial dimension. Set to pico_tree::kDynamicSize when the dimension is only
  // known at run-time.
  static std::size_t constexpr Dim = Dim_;

  // Returns the amount of coordinates of each point.
  inline static std::size_t SpaceSdim(SpaceType const&) { return Dim; }

  // Returns number of points contained by the space.
  inline static IndexType SpaceNpts(SpaceType const& space) {
    return static_cast<IndexType>(space.size());
  }

  // Returns the idx'th point from the input space.
  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space[idx];
  }

  // Returns the spatial dimension of the input point. Allowing the input type
  // to be different from PointType could give greater interfacing flexibility.
  inline static std::size_t PointSdim(PointType const& point) {
    return point.size();
  }

  // Returns a pointer to the coordinates of the input point. Allowing the input
  // type to be different from PointType could give greater interfacing
  // flexibility.
  inline static ScalarType const* PointCoords(PointType const& point) {
    return point.data();
  }
};

// This version of Traits makes the KdTree take the input by const reference.
template <typename Scalar_, std::size_t Dim_>
struct ConstRefTraits : public Traits<Scalar_, Dim_> {
  using SpaceType =
      std::reference_wrapper<typename Traits<Scalar_, Dim_>::SpaceType const>;
};

int main() {
  int max_leaf_size = 1;
  std::deque<std::array<float, 2>> points{
      {0.0f, 1.0f}, {2.0f, 3.0f}, {4.0f, 5.0f}};

  // The KdTree always takes the input by value. A copy can be prevented by
  // moving the points into the tree. Or use ConstRefTraits instead so that the
  // points will be taken by const reference.
  pico_tree::KdTree<Traits<float, 2>> tree(std::move(points), max_leaf_size);

  std::array<float, 2> query{4.0f, 4.0f};
  pico_tree::Neighbor<int, float> nn;
  tree.SearchNn(query, &nn);

  std::cout << "Index closest point: " << nn.index << std::endl;

  return 0;
}
