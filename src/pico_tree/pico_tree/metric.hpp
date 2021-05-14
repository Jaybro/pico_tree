#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! Mathematical constant pi. It is defined as the ratio of a circle's
//! circumference to its diameter. Only available from C++20.
static constexpr long double kPi = 3.14159265358979323846l;

//! \brief Calculates the absolute difference between two point coordinates.
struct AbsDiff {
  //! \private
  template <typename Scalar>
  inline static Scalar Op(Scalar x, Scalar y) {
    return std::abs(x - y);
  }
};

//! \brief Calculates the squared difference between two point coordinates.
struct SqrdDiff {
  //! \private
  template <typename Scalar>
  inline static Scalar Op(Scalar x, Scalar y) {
    Scalar const d = x - y;
    return d * d;
  }
};

//! \brief Calculates the difference between all coordinats of two points given
//! a binary operator.
//! \tparam Traits Interface for intput points.
//! \tparam BinOp Operator used to calculate coordinate differences.
template <typename Traits, typename BinOp>
struct Sum {
  //! \private
  using ScalarType = typename Traits::ScalarType;

  //! \private
  template <typename P0, typename P1>
  inline static ScalarType Op(P0 const& p0, P1 const& p1) {
    assert(Traits::PointSdim(p0) == Traits::PointSdim(p1));

    ScalarType const* c0 = Traits::PointCoords(p0);
    ScalarType const* c1 = Traits::PointCoords(p1);
    ScalarType d{};

    for (int i = 0; i < internal::Dimension<Traits>::Dim(p0); ++i) {
      d += BinOp::Op(c0[i], c1[i]);
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
//! \details Supports the fastest queries doesn't support identificatons.
//! \see TopologicalSpaceTag
class EuclideanSpaceTag : public TopologicalSpaceTag {};

//! \brief L1 metric for measuring the Taxicab or Manhattan distance between
//! points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
template <typename Traits>
class L1 {
 private:
  using Scalar = typename Traits::ScalarType;

 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(Scalar, Scalar).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      Scalar>::type
  operator()(P0 const& p0, P1 const& p1) const {
    return internal::Sum<Traits, internal::AbsDiff>::Op(p0, p1);
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }
};

//! \brief The L2Squared semimetric measures squared Euclidean distances between
//! points. It does not satisfy the triangle inequality.
//! \see L1
//! \see L2
template <typename Traits>
class L2Squared {
 private:
  using Scalar = typename Traits::ScalarType;

 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = EuclideanSpaceTag;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(Scalar, Scalar).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      Scalar>::type
  operator()(P0 const& p0, P1 const& p1) const {
    return internal::Sum<Traits, internal::SqrdDiff>::Op(p0, p1);
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared value of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }
};

//! \brief The SO2 metric measures distances on the unit circle S1. It is the
//! intrinsic metric of points in R2 on S1 given by the great-circel distance.
//! \details Named after the Special Orthogonal Group of dimension 2. The circle
//! S1 is represented by the range [-PI, PI] / -PI ~ PI.
//!
//! For more details:
//! * https://en.wikipedia.org/wiki/Intrinsic_metric
//! * https://en.wikipedia.org/wiki/Great-circle_distance
template <typename Traits>
class SO2 {
 private:
  using Scalar = typename Traits::ScalarType;
  static constexpr auto kTwoPi = static_cast<Scalar>(internal::kPi * 2.0l);

 public:
  //! \brief This tag specifies the supported space by this metric.
  using SpaceTag = TopologicalSpaceTag;

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(Scalar, Scalar).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      Scalar>::type
  operator()(P0 const& p0, P1 const& p1) const {
    assert(Traits::PointSdim(p0) == Traits::PointSdim(p1));
    assert(Traits::PointSdim(p0) == 1);

    return operator()(*Traits::PointCoords(p0), *Traits::PointCoords(p1));
  }

  //! \brief Calculates the distance between two coordinates.
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  inline Scalar operator()(Scalar const x, Scalar const y, int const) const {
    return operator()(x, y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }

  //! \brief Calculates the distance between \p x and the box defined by [ \p
  //! min, \p max ].
  //! \details The last argument is the dimension. It can be used to support
  //! Cartesian products of spaces but it is ignored here.
  inline Scalar operator()(
      Scalar const x, Scalar const min, Scalar const max, int const) const {
    // Rectangles currently can't be around the identification of PI ~ -PI where
    // the minimum is larger than he maximum.
    if (x < min || x > max) {
      return std::min(operator()(x, min), operator()(x, max));
    }

    return Scalar(0.0);
  }

 private:
  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = std::abs(x - y);
    return std::min(d, kTwoPi - d);
  }
};

}  // namespace pico_tree
