#pragma once

#include <random>

#include "matrix_space_traits.hpp"
#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/internal/space_wrapper.hpp"
#include "point_traits.hpp"

namespace pico_tree::internal {

template <typename Scalar_, Size Dim_>
inline Point<Scalar_, Dim_> RandomNormal(Size dim) {
  std::random_device rd;
  std::mt19937 e(rd());
  std::normal_distribution<Scalar_> gaussian(Scalar_(0), Scalar_(1));

  Point<Scalar_, Dim_> v = Point<Scalar_, Dim_>::FromSize(dim);

  for (Size i = 0; i < dim; ++i) {
    v[i] = gaussian(e);
  }

  v.Normalize();

  return v;
}

// Rotating datasets is computationally expensive. It is quadratic in the
// dimension of the space. Because we're only actually interested in obtaining a
// random orthogonal basis, any orthogonal transformation matrix will do.
// Householder matrices can be used to obtain linear complexity for rotating a
// dataset.
// C. Silpa-Anan and R. Hartley. Optimised KD-trees for fast image descriptor
// matching. In IEEE Conference on Computer Vision and Pattern Recognition,
// pages 1-8, 2008.
template <typename Node_, Size Dim_>
class RKdTreeHhData {
 public:
  using ScalarType = typename Node_::ScalarType;
  static Size constexpr Dim = Dim_;
  using NodeType = Node_;
  using RotationType = Point<ScalarType, Dim_>;
  using SpaceType = MatrixSpace<ScalarType, Dim_>;
  using SpaceWrapperType = SpaceWrapper<SpaceType>;

  template <typename SpaceWrapper_>
  static inline auto RandomRotation(SpaceWrapper_ space) {
    return RandomNormal<ScalarType, Dim_>(space.sdim());
  }

  template <typename SpaceWrapper_>
  static inline SpaceType RotateSpace(
      RotationType const& rotation, SpaceWrapper_ space) {
    SpaceType s(space.size(), space.sdim());

    for (std::size_t i = 0; i < space.size(); ++i) {
      auto x = space[i];
      auto y = s.data(i);
      RotatePoint(rotation, x, y);
    }

    return s;
  }

  template <typename ArrayType_>
  Point<ScalarType, Dim_> RotatePoint(ArrayType_ const& x) const {
    Point<ScalarType, Dim_> y = Point<ScalarType, Dim_>::FromSize(space.sdim());
    RotatePoint(rotation, x, y);
    return y;
  }

  RotationType rotation;
  SpaceType space;
  KdTreeData<Node_, Dim_> tree;

 private:
  // In and out can be the same point.
  // https://en.wikipedia.org/wiki/Householder_transformation
  template <typename ArrayTypeIn_, typename ArrayTypeOut_>
  static void RotatePoint(
      RotationType const& rotation, ArrayTypeIn_ const& x, ArrayTypeOut_& y) {
    ScalarType dot = ScalarType(0);
    for (Size i = 0; i < rotation.size(); ++i) {
      dot += rotation[i] * x[i];
    }
    dot *= ScalarType(2);
    for (Size i = 0; i < rotation.size(); ++i) {
      y[i] = x[i] - (dot * rotation[i]);
    }
  }
};

}  // namespace pico_tree::internal
