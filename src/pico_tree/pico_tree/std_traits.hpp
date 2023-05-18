#pragma once

#include <functional>
#include <vector>

#include "core.hpp"
#include "point_traits.hpp"

namespace pico_tree {

//! \brief StdTraits provides an interface for spaces and points when working
//! with indexable containers from the C++ standard.
//! \details Because different point types can have different interfaces, they
//! will be provided by PointTraits.
//! \tparam Space_ Any of the point sets supported by PointTraits.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Space_, typename Index_ = int>
struct StdTraits;

//! \brief StdTraits provides an interface for std::vector<> and points
//! supported by PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Point_, typename Allocator_, typename Index_>
struct StdTraits<std::vector<Point_, Allocator_>, Index_> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::vector<Point_, Allocator_>;
  //! \brief The point type used by SpaceType.
  using PointType = Point_;
  //! \brief The scalar type of point coordinates.
  using ScalarType = typename PointTraits<Point_>::ScalarType;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief The index type of point coordinates.
  using IndexType = Index_;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = PointTraits<Point_>::Dim;

  static_assert(
      Dim != kDynamicSize, "VECTOR_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType constexpr SpaceSdim(
      std::vector<Point_, Allocator_> const&) {
    return Dim;
  }

  //! \brief Returns number of points contained by \p space.
  inline static IndexType SpaceNpts(
      std::vector<Point_, Allocator_> const& space) {
    return static_cast<IndexType>(space.size());
  }

  //! \brief Returns the point at \p idx from \p space.
  inline static Point_ const& PointAt(
      std::vector<Point_, Allocator_> const& space, IndexType const idx) {
    return space[idx];
  }

  //! \brief Returns the spatial dimension of \p point.
  //! \details Allowing the input type to be different from PointType gives us
  //! greater interfacing flexibility.
  template <typename OtherPoint>
  inline static SizeType PointSdim(OtherPoint const& point) {
    return PointTraits<OtherPoint>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  //! \details Allowing the input type to be different from PointType gives us
  //! greater interfacing flexibility.
  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return PointTraits<OtherPoint>::Coords(point);
  }
};

//! \brief StdTraits provides an interface for
//! std::reference_wrapper<std::vector<>> and points supported by
//! PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Point_, typename Allocator_, typename Index_>
struct StdTraits<
    std::reference_wrapper<std::vector<Point_, Allocator_>>,
    Index_> : public StdTraits<std::vector<Point_, Allocator_>, Index_> {
  //! \brief The SpaceType of these traits.
  //! \details This overrides the SpaceType of the base class. No other
  //! interface changes are required as an std::reference_wrapper can implicitly
  //! be converted to its contained reference, which is a reference to an object
  //! of the exact same type as that of the SpaceType of the base class.
  using SpaceType = std::reference_wrapper<std::vector<Point_, Allocator_>>;
};

//! \brief StdTraits provides an interface for
//! std::reference_wrapper<std::vector<> const> and points supported by
//! PointTraits.
//! \tparam Point_ Any of the point types supported by PointTraits.
//! \tparam Allocator_ Allocator type used by the std::vector.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Point_, typename Allocator_, typename Index_>
struct StdTraits<
    std::reference_wrapper<std::vector<Point_, Allocator_> const>,
    Index_> : public StdTraits<std::vector<Point_, Allocator_>, Index_> {
  //! \brief The SpaceType of these traits.
  //! \details This overrides the SpaceType of the base class. No other
  //! interface changes are required as an std::reference_wrapper can implicitly
  //! be converted to its contained reference, which is a reference to an object
  //! of the exact same type as that of the SpaceType of the base class.
  using SpaceType =
      std::reference_wrapper<std::vector<Point_, Allocator_> const>;
};

}  // namespace pico_tree
