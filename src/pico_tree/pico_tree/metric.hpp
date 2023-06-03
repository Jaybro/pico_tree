#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! Mathematical constant pi. It is defined as the ratio of a circle's
//! circumference to its diameter. Only available from C++20.
template <typename T>
inline T constexpr kPi = T(3.14159265358979323846l);
template <typename T>
inline T constexpr kTwoPi = T(6.28318530717958647693l);

//! \brief Calculates the absolute difference between two point coordinates.
struct AbsDiff {
  //! \private
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return std::abs(x - y);
  }
};

//! \brief Calculates the squared difference between two point coordinates.
struct SqrdDiff {
  //! \private
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    Scalar_ const d = x - y;
    return d * d;
  }
};

//! \brief Calculates the distance between two angles.
//! \details The circle S1 is represented by the range [-PI, PI] / -PI ~ PI.
struct AngleAbsDiff {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    Scalar_ const d = std::abs(x - y);
    return std::min(d, internal::kTwoPi<Scalar_> - d);
  }
};

template <
    typename InputIterator1,
    typename InputSentinel1,
    typename OutputIterator2,
    typename BinaryOperator>
constexpr auto Sum(
    InputIterator1 begin1,
    InputSentinel1 end1,
    OutputIterator2 begin2,
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
class L1 {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename OutputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1,
      InputSentinel1 end1,
      OutputIterator2 begin2) const {
    return internal::Sum(begin1, end1, begin2, internal::AbsDiff());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x, Scalar_ const y) const {
    return internal::AbsDiff()(x, y);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x) const {
    return std::abs(x);
  }
};

//! \brief The L2Squared semimetric measures squared Euclidean distances between
//! points. It does not satisfy the triangle inequality.
//! \see L1
class L2Squared {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename OutputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1,
      InputSentinel1 end1,
      OutputIterator2 begin2) const {
    return internal::Sum(begin1, end1, begin2, internal::SqrdDiff());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x, Scalar_ const y) const {
    return internal::SqrdDiff()(x, y);
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x) const {
    return x * x;
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
class SO2 {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = TopologicalSpaceTag;

  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename OutputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1, InputSentinel1, OutputIterator2 begin2) const {
    return operator()(*begin1, *begin2);
  }

  //! \brief Calculates the distance between \p x and the box defined by [ \p
  //! min, \p max ].
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  template <typename Scalar_>
  constexpr Scalar_ operator()(
      Scalar_ const x, Scalar_ const min, Scalar_ const max, int const) const {
    // Rectangles can't currently wrap around the identification of PI ~ -PI
    // where the minimum is larger than he maximum.
    if (x < min || x > max) {
      return std::min(operator()(x, min), operator()(x, max));
    }

    return Scalar_(0.0);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x) const {
    return std::abs(x);
  }

 private:
  template <typename Scalar_>
  constexpr auto operator()(Scalar_ x, Scalar_ y) const {
    return internal::AngleAbsDiff()(x, y);
  }
};

}  // namespace pico_tree
