#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! Mathematical constant pi. It is defined as the ratio of a circle's
//! circumference to its diameter. Only available from C++20.
static long double constexpr kPi = 3.14159265358979323846l;
static long double constexpr kTwoPi = kPi * 2.0l;

//! \brief Calculates the absolute difference between two point coordinates.
struct AbsDiff {
  //! \private
  template <typename Scalar_>
  inline static Scalar_ Op(Scalar_ x, Scalar_ y) {
    return std::abs(x - y);
  }
};

//! \brief Calculates the squared difference between two point coordinates.
struct SqrdDiff {
  //! \private
  template <typename Scalar_>
  inline static Scalar_ Op(Scalar_ x, Scalar_ y) {
    Scalar_ const d = x - y;
    return d * d;
  }
};

//! \brief Calculates the difference between all coordinats of two points given
//! a binary operator.
//! \tparam Traits_ Interface for intput points.
//! \tparam BinOp_ Operator used to calculate coordinate differences.
template <typename Traits_, typename BinOp_>
struct Sum {
  //! \private
  using ScalarType = typename Traits_::ScalarType;

  //! \private
  template <typename P0, typename P1>
  inline static ScalarType Op(P0 const& p0, P1 const& p1) {
    assert(Traits_::PointSdim(p0) == Traits_::PointSdim(p1));

    ScalarType const* c0 = Traits_::PointCoords(p0);
    ScalarType const* c1 = Traits_::PointCoords(p1);
    ScalarType d{};

    for (Size i = 0; i < internal::Dimension<Traits_>::Dim(p0); ++i) {
      d += BinOp_::Op(c0[i], c1[i]);
    }

    return d;
  }
};

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
template <typename Traits_>
class L1 {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;
  using ScalarType = typename Traits_::ScalarType;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(ScalarType, ScalarType).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      ScalarType>::type
  operator()(P0 const& p0, P1 const& p1) const {
    return internal::Sum<Traits_, internal::AbsDiff>::Op(p0, p1);
  }

  //! \brief Calculates the distance between two coordinates.
  inline ScalarType operator()(ScalarType const x, ScalarType const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline ScalarType operator()(ScalarType const x) const { return std::abs(x); }
};

//! \brief The L2Squared semimetric measures squared Euclidean distances between
//! points. It does not satisfy the triangle inequality.
//! \see L1
//! \see L2
template <typename Traits_>
class L2Squared {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;
  using ScalarType = typename Traits_::ScalarType;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(ScalarType, ScalarType).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      ScalarType>::type
  operator()(P0 const& p0, P1 const& p1) const {
    return internal::Sum<Traits_, internal::SqrdDiff>::Op(p0, p1);
  }

  //! \brief Calculates the distance between two coordinates.
  inline ScalarType operator()(ScalarType const x, ScalarType const y) const {
    ScalarType const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared value of \p x.
  inline ScalarType operator()(ScalarType const x) const { return x * x; }
};

//! \brief The SO2 metric measures distances on the unit circle S1. It is the
//! intrinsic metric of points in R2 on S1 given by the great-circel distance.
//! \details Named after the Special Orthogonal Group of dimension 2. The circle
//! S1 is represented by the range [-PI, PI] / -PI ~ PI.
//!
//! For more details:
//! * https://en.wikipedia.org/wiki/Intrinsic_metric
//! * https://en.wikipedia.org/wiki/Great-circle_distance
template <typename Traits_>
class SO2 {
 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = TopologicalSpaceTag;
  using ScalarType = typename Traits_::ScalarType;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(ScalarType, ScalarType).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      ScalarType>::type
  operator()(P0 const& p0, P1 const& p1) const {
    assert(Traits_::PointSdim(p0) == Traits_::PointSdim(p1));
    assert(Traits_::PointSdim(p0) == 1);

    return operator()(*Traits_::PointCoords(p0), *Traits_::PointCoords(p1));
  }

  //! \brief Returns the absolute value of \p x.
  inline ScalarType operator()(ScalarType const x) const { return std::abs(x); }

  //! \brief Calculates the distance between \p x and the box defined by [ \p
  //! min, \p max ].
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  inline ScalarType operator()(
      ScalarType const x,
      ScalarType const min,
      ScalarType const max,
      int const) const {
    // Rectangles can't currently wrap around the identification of PI ~ -PI
    // where the minimum is larger than he maximum.
    if (x < min || x > max) {
      return std::min(operator()(x, min), operator()(x, max));
    }

    return ScalarType(0.0);
  }

 private:
  static ScalarType constexpr kTwoPi =
      static_cast<ScalarType>(internal::kTwoPi);

  //! \brief Calculates the distance between two coordinates.
  inline ScalarType operator()(ScalarType const x, ScalarType const y) const {
    ScalarType const d = std::abs(x - y);
    return std::min(d, kTwoPi - d);
  }
};

}  // namespace pico_tree
