#pragma once

#include "pico_tree/map_traits.hpp"
#include "pico_understory/internal/matrix_space.hpp"

namespace pico_tree {

template <typename Scalar_, size_t Dim_>
struct space_traits<internal::matrix_space<Scalar_, Dim_>> {
  using space_type = internal::matrix_space<Scalar_, Dim_>;
  using point_type = point_map<Scalar_ const, Dim_>;
  using scalar_type = typename space_type::scalar_type;
  using size_type = typename space_type::size_type;
  static size_type constexpr dim = space_type::dim;

  template <typename Index_>
  inline static point_type point_at(space_type const& space, Index_ idx) {
    return space[static_cast<size_type>(idx)];
  }

  inline static size_type size(space_type const& space) { return space.size(); }

  inline static size_type sdim(space_type const& space) { return space.sdim(); }
};

}  // namespace pico_tree
