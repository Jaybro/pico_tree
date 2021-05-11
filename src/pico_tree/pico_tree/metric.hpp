#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

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
class TopologicalSpaceTag {};

//! \brief Identifies a metric to support the Euclidean space with PicoTree's
//! search structures.
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

}  // namespace pico_tree
