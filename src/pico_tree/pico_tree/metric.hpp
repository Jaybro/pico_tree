#pragma once

#include "core.hpp"

namespace pico_tree {

//! \brief L1 metric for measuring the Taxicab or Manhattan distance between
//! points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
template <typename Scalar, int Dim>
class L1 {
 public:
  //! \brief Creates an L1 metric given a spatial dimension.
  inline explicit L1(int const dim) : dim_{dim} {}

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
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(dim_); ++i) {
      d += std::abs(p0(i) - p1(i));
    }

    return d;
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }

 private:
  int const dim_;
};

//! \brief L2 metric for measuring Euclidean distances between points.
//! \details https://en.wikipedia.org/wiki/Euclidean_distance
//! \see L1
template <typename Scalar, int Dim>
class L2 {
 public:
  //! \brief Creates an L2Squared given a spatial dimension.
  inline explicit L2(int const dim) : dim_{dim} {}

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
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(dim_); ++i) {
      Scalar const v = p0(i) - p1(i);
      d += v * v;
    }

    return std::sqrt(d);
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }

 private:
  int const dim_;
};

//! \brief The L2Squared semimetric measures squared Euclidean distances between
//! points. It does not satisfy the triangle inequality.
//! \see L1
//! \see L2
template <typename Scalar, int Dim>
class L2Squared {
 public:
  //! \brief Creates an L2Squared given a spatial dimension.
  inline explicit L2Squared(int const dim) : dim_{dim} {}

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
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(dim_); ++i) {
      Scalar const v = p0(i) - p1(i);
      d += v * v;
    }

    return d;
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared value of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }

 private:
  int const dim_;
};

}  // namespace pico_tree
