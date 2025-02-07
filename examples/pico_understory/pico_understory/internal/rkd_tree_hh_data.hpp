#pragma once

#include <random>

#include "pico_tree/internal/kd_tree_data.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/internal/space_wrapper.hpp"
#include "pico_understory/internal/matrix_space_traits.hpp"
#include "point_traits.hpp"

namespace pico_tree::internal {

template <typename Scalar_, size_t Dim_>
inline point<Scalar_, Dim_> random_normal(size_t dim) {
  std::random_device rd;
  std::mt19937 e(rd());
  std::normal_distribution<Scalar_> gaussian(Scalar_(0), Scalar_(1));

  point<Scalar_, Dim_> v = point<Scalar_, Dim_>::from_size(dim);

  for (size_t i = 0; i < dim; ++i) {
    v[i] = gaussian(e);
  }

  v.normalize();

  return v;
}

// Rotating datasets is computationally expensive. It is quadratic in the
// dimension of the space. Because we're only actually interested in obtaining a
// random orthogonal basis, any orthogonal transformation matrix will do, such
// as the Householder matrix. Householder matrices can be used to obtain linear
// complexity for "rotating" a dataset. C. Silpa-Anan and R. Hartley, Optimised
// KD-trees for fast image descriptor matching, In CVPR, 2008.
// http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_CVPR_2008/data/papers/298.pdf
template <typename Node_, size_t Dim_>
class rkd_tree_hh_data {
 public:
  using scalar_type = typename Node_::scalar_type;
  static size_t constexpr dim = Dim_;
  using node_type = Node_;
  using rotation_type = point<scalar_type, Dim_>;
  using space_type = matrix_space<scalar_type, Dim_>;
  using space_wrapper_type = space_wrapper<space_type>;

  template <typename SpaceWrapper_>
  static inline auto random_rotation(SpaceWrapper_ space) {
    return random_normal<scalar_type, Dim_>(space.sdim());
  }

  template <typename SpaceWrapper_>
  static inline space_type rotate_space(
      rotation_type const& rotation, SpaceWrapper_ space) {
    space_type s(space.size(), space.sdim());

    for (std::size_t i = 0; i < space.size(); ++i) {
      auto x = space[i];
      auto y = s.data(i);
      rotate_point(rotation, x, y);
    }

    return s;
  }

  template <typename ArrayType_>
  point<scalar_type, Dim_> rotate_point(ArrayType_ const& x) const {
    point<scalar_type, Dim_> y =
        point<scalar_type, Dim_>::from_size(space.sdim());
    rotate_point(rotation, x, y);
    return y;
  }

  rotation_type rotation;
  space_type space;
  kd_tree_data<Node_, Dim_> tree;

 private:
  // In and out can be the same point.
  // https://en.wikipedia.org/wiki/Householder_transformation
  template <typename ArrayTypeIn_, typename ArrayTypeOut_>
  static void rotate_point(
      rotation_type const& rotation, ArrayTypeIn_ const& x, ArrayTypeOut_& y) {
    scalar_type dot = scalar_type(0);
    for (size_t i = 0; i < rotation.size(); ++i) {
      dot += rotation[i] * x[i];
    }
    dot *= scalar_type(2);
    for (size_t i = 0; i < rotation.size(); ++i) {
      y[i] = x[i] - (dot * rotation[i]);
    }
  }
};

}  // namespace pico_tree::internal
