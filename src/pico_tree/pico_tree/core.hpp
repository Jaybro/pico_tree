#pragma once

//! \mainpage PicoTree is a C++ header only library for nearest neighbor
//! searches and range searches using a KdTree.
//! \file core.hpp
//! \brief Contains various common utilities.

#include <type_traits>

namespace pico_tree {

//! \brief Size type used by PicoTree.
using Size = std::size_t;

//! \brief This value can be used in any template argument that wants to know
//! the spatial dimension of the search problem when it can only be known at
//! run-time. In this case the dimension of the problem is provided by the point
//! adaptor.
inline Size constexpr kDynamicSize = static_cast<Size>(-1);

//! \brief A Neighbor is a point reference with a corresponding distance to
//! another point.
template <typename Index_, typename Scalar_>
struct Neighbor {
  static_assert(std::is_integral_v<Index_>, "INDEX_NOT_AN_INTEGRAL_TYPE");
  static_assert(std::is_arithmetic_v<Scalar_>, "SCALAR_NOT_AN_ARITHMETIC_TYPE");

  //! \brief Index type.
  using IndexType = Index_;
  //! \brief Distance type.
  using ScalarType = Scalar_;

  //! \brief Default constructor.
  //! \details Declaring a custom constructor removes the default one. With
  //! C++11 we can bring back the default constructor and keep this struct a POD
  //! type.
  constexpr Neighbor() = default;
  //! \brief Constructs a Neighbor given an index and distance.
  constexpr Neighbor(IndexType idx, ScalarType dst) noexcept
      : index(idx), distance(dst) {}

  //! \brief Point index of the Neighbor.
  IndexType index;
  //! \brief Distance of the Neighbor with respect to another point.
  ScalarType distance;
};

//! \brief Compares neighbors by distance.
template <typename Index_, typename Scalar_>
constexpr bool operator<(
    Neighbor<Index_, Scalar_> const& lhs,
    Neighbor<Index_, Scalar_> const& rhs) noexcept {
  return lhs.distance < rhs.distance;
}

}  // namespace pico_tree
