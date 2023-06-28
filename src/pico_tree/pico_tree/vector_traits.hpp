#pragma once

#include <vector>

#include "core.hpp"
#include "point_traits.hpp"
#include "space_traits.hpp"

namespace pico_tree {

//! \brief Provides an interface for std::vector<> and points supported by
//! PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
template <typename Point_, typename Allocator_>
struct SpaceTraits<std::vector<Point_, Allocator_>> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::vector<Point_, Allocator_>;
  //! \brief The point type used by SpaceType.
  using PointType = Point_;
  //! \brief The scalar type of point coordinates.
  using ScalarType = typename PointTraits<Point_>::ScalarType;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = PointTraits<Point_>::Dim;

  static_assert(
      Dim != kDynamicSize, "VECTOR_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType constexpr Sdim(SpaceType const&) { return Dim; }

  //! \brief Returns number of points contained by \p space.
  inline static SizeType Npts(SpaceType const& space) { return space.size(); }

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static Point_ const& PointAt(SpaceType const& space, Index_ idx) {
    return space[static_cast<SizeType>(idx)];
  }
};

}  // namespace pico_tree
