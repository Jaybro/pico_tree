#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! \brief Mathematical constant pi. It is defined as the ratio of a circle's
//! circumference to its diameter. Only available from C++20.
template <typename T>
inline T constexpr kPi = T(3.14159265358979323846l);
template <typename T>
inline T constexpr kTwoPi = T(6.28318530717958647693l);

//! \brief Calculates the square of a number.
template <typename Scalar_>
constexpr Scalar_ Squared(Scalar_ x) {
  return x * x;
}

//! \brief Calculates the distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ Distance(Scalar_ x, Scalar_ y) {
  return std::abs(x - y);
}

//! \brief Calculates the distance between two coordinates.
struct DistanceFn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return Distance(x, y);
  }
};

//! \brief Calculates the squared distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ SquaredDistance(Scalar_ x, Scalar_ y) {
  return Squared(x - y);
}

//! \brief Calculates the squared distance between two coordinates.
struct SquaredDistanceFn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return SquaredDistance(x, y);
  }
};

//! \brief Calculates the distance between coordinate \p x and the box defined
//! by [ \p min, \p max ].
template <typename Scalar_>
constexpr Scalar_ DistanceBox(Scalar_ x, Scalar_ min, Scalar_ max) {
  if (x < min) {
    return min - x;
  } else if (x > max) {
    return x - max;
  } else {
    return Scalar_(0.0);
  }
}

//! \brief Calculates the squared distance between coordinate \p x and the box
//! defined by [ \p min, \p max ].
template <typename Scalar_>
constexpr Scalar_ SquaredDistanceBox(Scalar_ x, Scalar_ min, Scalar_ max) {
  return Squared(DistanceBox(x, min, max));
}

//! \brief Calculates the angular distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ AngleDistance(Scalar_ x, Scalar_ y) {
  Scalar_ const d = std::abs(x - y);
  return std::min(d, internal::kTwoPi<Scalar_> - d);
}

//! \brief Calculates the squared angular distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ SquaredAngleDistance(Scalar_ x, Scalar_ y) {
  return Squared(AngleDistance(x, y));
}

//! \brief Calculates the angular distance between coordinate \p x and the box
//! defined by [ \p min, \p max ].
template <typename Scalar_>
constexpr Scalar_ AngleDistanceBox(Scalar_ x, Scalar_ min, Scalar_ max) {
  // Rectangles can't currently wrap around the identification of PI ~ -PI
  // where the minimum is larger than he maximum.
  if (x < min || x > max) {
    return std::min(AngleDistance(x, min), AngleDistance(x, max));
  } else {
    return Scalar_(0.0);
  }
}

//! \brief Calculates the squared angular distance between a coordinate and a
//! box.
template <typename Scalar_>
constexpr Scalar_ SquaredAngleDistanceBox(Scalar_ x, Scalar_ min, Scalar_ max) {
  return Squared(AngleDistance(x, min, max));
}

//! \brief Calculates the squared angular distance between two coordinates.
//! \details The circle S1 is represented by the range [-PI, PI] / -PI ~ PI.
struct AngleDistanceFn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return AngleDistance(x, y);
  }
};

template <
    typename InputIterator1,
    typename InputSentinel1,
    typename InputIterator2,
    typename BinaryOperator>
constexpr auto Sum(
    InputIterator1 begin1,
    InputSentinel1 end1,
    InputIterator2 begin2,
    BinaryOperator op) {
  using ScalarType = typename std::iterator_traits<InputIterator1>::value_type;

  ScalarType d{};

  for (; begin1 != end1; ++begin1, ++begin2) {
    d += op(*begin1, *begin2);
  }

  return d;
}

}  // namespace internal

//! \brief Identifies a metric to support the most generic space that can be
//! used with PicoTree's search structures.
//! \details A space tag is used by PicoTree to select the most optimal
//! algorithms for use with a particular space.
//!
//! Usings the TopologicalSpaceTag for metrics allows support for
//! identifications in point sets. A practical example is that of the unit
//! circle represented by the interval [-PI, PI]. Here, -PI and PI are the same
//! point on the circle and performing a radius query around both values should
//! result in the same point set.
class TopologicalSpaceTag {};

//! \brief Identifies a metric to support the Euclidean space with PicoTree's
//! search structures.
//! \details Supports the fastest queries but doesn't support identificatons.
//! \see TopologicalSpaceTag
class EuclideanSpaceTag : public TopologicalSpaceTag {};

//! \brief L1 metric for measuring the Taxicab or Manhattan distance between
//! points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
struct L1 {
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename InputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1, InputSentinel1 end1, InputIterator2 begin2) const {
    return internal::Sum(begin1, end1, begin2, internal::DistanceFn());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return internal::Distance(x, y);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The L2Squared semimetric measures squared Euclidean distances between
//! points. It does not satisfy the triangle inequality.
//! \see L1
struct L2Squared {
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename InputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1, InputSentinel1 end1, InputIterator2 begin2) const {
    return internal::Sum(begin1, end1, begin2, internal::SquaredDistanceFn());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return internal::SquaredDistance(x, y);
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return internal::Squared(x);
  }
};

//! \brief The SO2 metric measures distances on the unit circle S1. It is the
//! intrinsic metric of points in R2 on S1 given by the great-circel distance.
//! \details Named after the Special Orthogonal Group of dimension 2. The circle
//! S1 is represented by the range [-PI, PI] / -PI ~ PI.
//!
//! For more details:
//! * https://en.wikipedia.org/wiki/Intrinsic_metric
//! * https://en.wikipedia.org/wiki/Great-circle_distance
struct SO2 {
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = TopologicalSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename InputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1, InputSentinel1, InputIterator2 begin2) const {
    return internal::AngleDistance(*begin1, *begin2);
  }

  //! \brief Calculates the distance between coordinate \p x and the box defined
  //! by [ \p min, \p max ].
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ min, Scalar_ max, int) const {
    return internal::AngleDistanceBox(x, min, max);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The SE2Squared metric measures distances in Euclidean space between
//! Euclidean motions.
//! \details Named after the Special Euclidean group of dimension 2.
//! For more details:
//! * https://en.wikipedia.org/wiki/Euclidean_group
struct SE2Squared {
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = TopologicalSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename InputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1, InputSentinel1, InputIterator2 begin2) const {
    return internal::Sum(
               begin1, begin1 + 2, begin2, internal::SquaredDistanceFn()) +
           internal::SquaredAngleDistance(*(begin1 + 2), *(begin2 + 2));
  }

  //! \brief Calculates the squared distance between coordinate \p x and the box
  //! defined by [ \p min, \p max ].
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  template <typename Scalar_>
  constexpr Scalar_ operator()(
      Scalar_ x, Scalar_ min, Scalar_ max, int dim) const {
    if (dim < 2) {
      return internal::SquaredDistanceBox(x, min, max);
    } else {
      return internal::SquaredAngleDistanceBox(x, min, max);
    }
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return internal::Squared(x);
  }
};

}  // namespace pico_tree
