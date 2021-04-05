#pragma once

#include <functional>
#include <vector>

#include "core.hpp"

namespace pico_tree {

//! \brief StdTraits provides an interface for spaces and points when working
//! with indexable containers from the C++ standard.
//! \details Because different point types can have different interfaces, they
//! will be provided by StdPointTraits.
//! \tparam Space Any of the point sets supported by StdPointTraits.
//! \tparam Index Type used for indexing. Defaults to int.
template <typename Space, typename Index = int>
struct StdTraits;

//! \brief StdPointTraits provides an interface for the different point types
//! that can be used with StdTraits (or others).
//! \tparam Point Any of the point types supported by StdPointTraits.
template <typename Point>
struct StdPointTraits;

//! \brief StdTraits provides an interface for std::vector<> and points
//! supported by StdPointTraits.
//! \tparam Point Any of the point types supported by StdPointTraits.
//! \tparam Allocator Allocator type for the std::vector.
//! \tparam Index Type used for indexing. Defaults to int.
template <typename Point, typename Allocator, typename Index>
struct StdTraits<std::vector<Point, Allocator>, Index> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::vector<Point, Allocator>;
  //! \brief The point type used by SpaceType.
  using PointType = Point;
  //! \brief The scalar type of point coordinates.
  using ScalarType = typename StdPointTraits<Point>::ScalarType;
  //! \brief The index type of point coordinates.
  using IndexType = Index;
  //! \brief Compile time spatial dimension.
  static constexpr int Dim = StdPointTraits<Point>::Dim;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int constexpr SpaceSdim(std::vector<Point, Allocator> const&) {
    static_assert(
        Dim != kDynamicDim, "VECTOR_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");
    return Dim;
  }

  //! \brief Returns number of points contained by \p space.
  inline static IndexType SpaceNpts(
      std::vector<Point, Allocator> const& space) {
    return static_cast<IndexType>(space.size());
  }

  //! \brief Returns the point at \p idx from \p space.
  inline static Point const& PointAt(
      std::vector<Point, Allocator> const& space, IndexType const idx) {
    return space[idx];
  }

  //! \brief Returns the spatial dimension of \p point.
  template <typename OtherPoint>
  inline static int PointSdim(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Coords(point);
  }
};

//! \brief StdTraits provides an interface for
//! std::reference_wrapper<std::vector<>> and points supported by
//! StdPointTraits.
//! \tparam Point Any of the point types supported by StdPointTraits.
//! \tparam Allocator Allocator type for the std::vector.
//! \tparam Index Type used for indexing. Defaults to int.
template <typename Point, typename Allocator, typename Index>
struct StdTraits<std::reference_wrapper<std::vector<Point, Allocator>>, Index>
    : public StdTraits<std::vector<Point, Allocator>, Index> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::reference_wrapper<std::vector<Point, Allocator>>;
};

//! \brief StdTraits provides an interface for
//! std::reference_wrapper<std::vector<> const> and points supported by
//! StdPointTraits.
//! \tparam Point Any of the point types supported by StdPointTraits.
//! \tparam Allocator Allocator type for the std::vector.
//! \tparam Index Type used for indexing. Defaults to int.
template <typename Point, typename Allocator, typename Index>
struct StdTraits<
    std::reference_wrapper<std::vector<Point, Allocator> const>,
    Index> : public StdTraits<std::vector<Point, Allocator>, Index> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::reference_wrapper<std::vector<Point, Allocator> const>;
};

}  // namespace pico_tree
