#pragma once

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! \brief Mathematical constant pi. It is defined as the ratio of a circle's
//! circumference to its diameter. Only available from C++20.
template <typename T_>
inline T_ constexpr pi = T_(3.14159265358979323846l);
template <typename T_>
inline T_ constexpr two_pi = T_(6.28318530717958647693l);

//! \brief Calculates the square of a number.
template <typename Scalar_>
constexpr Scalar_ squared(Scalar_ x) {
  return x * x;
}

//! \brief Calculates the distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ distance(Scalar_ x, Scalar_ y) {
  return std::abs(x - y);
}

//! \brief Calculates the distance between two coordinates.
struct distance_fn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return distance(x, y);
  }
};

//! \brief Calculates the squared distance between two coordinates.
template <typename Scalar_>
constexpr Scalar_ squared_distance(Scalar_ x, Scalar_ y) {
  return squared(x - y);
}

//! \brief Calculates the squared distance between two coordinates.
struct squared_distance_fn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return squared_distance(x, y);
  }
};

//! \brief Calculates the distance between coordinate \p x and the box defined
//! by [ \p min, \p max ].
template <typename Scalar_>
constexpr Scalar_ distance_box(Scalar_ x, Scalar_ min, Scalar_ max) {
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
constexpr Scalar_ squared_distance_box(Scalar_ x, Scalar_ min, Scalar_ max) {
  return squared(distance_box(x, min, max));
}

//! \brief Calculates the angular distance between two coordinates. The values
//! for \p x or \py y must lie within the range of [-pi...pi].
template <typename Scalar_>
constexpr Scalar_ angle_distance(Scalar_ x, Scalar_ y) {
  Scalar_ const d = std::abs(x - y);
  return std::min(d, internal::two_pi<Scalar_> - d);
}

//! \brief Calculates the squared angular distance between two coordinates.
//! \see angle_distance
template <typename Scalar_>
constexpr Scalar_ squared_angle_distance(Scalar_ x, Scalar_ y) {
  return squared(angle_distance(x, y));
}

//! \brief Calculates the angular distance between coordinate \p x and the box
//! defined by [ \p min, \p max ].
template <typename Scalar_>
constexpr Scalar_ angle_distance_box(Scalar_ x, Scalar_ min, Scalar_ max) {
  // The box of a kd_tree node cannot wrap around the identification of PI ~
  // -PI. This means we don't have to check if the minimum is larger than the
  // maximum to see which range is inside the box.
  if (x < min || x > max) {
    return std::min(angle_distance(x, min), angle_distance(x, max));
  } else {
    return Scalar_(0.0);
  }
}

//! \brief Calculates the squared angular distance between a coordinate and a
//! box.
template <typename Scalar_>
constexpr Scalar_ squared_angle_distance_box(
    Scalar_ x, Scalar_ min, Scalar_ max) {
  return squared(angle_distance_box(x, min, max));
}

//! \brief Calculates the squared angular distance between two coordinates.
//! \details The circle S1 is represented by the range [-PI, PI] / -PI ~ PI.
struct angle_distance_fn {
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return angle_distance(x, y);
  }
};

template <
    typename InputIterator1_,
    typename InputSentinel1_,
    typename InputIterator2_,
    typename BinaryOperator>
constexpr auto sum(
    InputIterator1_ begin1,
    InputSentinel1_ end1,
    InputIterator2_ begin2,
    BinaryOperator op) {
  using scalar_type =
      typename std::iterator_traits<InputIterator1_>::value_type;

  scalar_type d{};

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
//! Usings the topological_space_tag for metrics allows support for
//! identifications in point sets. A practical example is that of the unit
//! circle represented by the interval [-PI, PI]. Here, -PI and PI are the same
//! point on the circle and performing a radius query around both values should
//! result in the same point set.
class topological_space_tag {};

//! \brief Identifies a metric to support the Euclidean space with PicoTree's
//! search structures.
//! \details Supports the fastest queries but doesn't support identifications.
//! \see topological_space_tag
class euclidean_space_tag : public topological_space_tag {};

//! \brief metric_l1 metric for measuring the Taxicab or Manhattan distance
//! between points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
struct metric_l1 {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    return internal::sum(begin1, end1, begin2, internal::distance_fn());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return internal::distance(x, y);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The metric_l2_squared semi-metric measures squared Euclidean
//! distances between points. It does not satisfy the triangle inequality.
//! \see metric_l1
struct metric_l2_squared {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    return internal::sum(begin1, end1, begin2, internal::squared_distance_fn());
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return internal::squared_distance(x, y);
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return internal::squared(x);
  }
};

struct metric_linf {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = euclidean_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      InputIterator2_ begin2) const {
    using scalar_type =
        typename std::iterator_traits<InputIterator1_>::value_type;

    scalar_type d{};

    for (; begin1 != end1; ++begin1, ++begin2) {
      d = std::max(d, internal::distance(*begin1, *begin2));
    }

    return d;
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x, Scalar_ y) const {
    return internal::distance(x, y);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The metric_so2 measures distances on the unit circle S1. It is the
//! intrinsic metric of points in R2 on S1 given by the great-circle distance.
//! \details Named after the Special Orthogonal Group of dimension 2. The circle
//! S1 is represented by the range [-PI, PI] / -PI ~ PI.
//!
//! For more details:
//! * https://en.wikipedia.org/wiki/Intrinsic_metric
//! * https://en.wikipedia.org/wiki/Great-circle_distance
struct metric_so2 {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = topological_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1, InputSentinel1_, InputIterator2_ begin2) const {
    return internal::angle_distance(*begin1, *begin2);
  }

  //! \brief Calculates the distance between coordinate \p x and the box defined
  //! by [ \p min, \p max ].
  //! \details The dimension argument can be used to support Cartesian products
  //! of spaces but it is ignored here.
  //! \see metric_se2_squared
  template <typename Scalar_>
  constexpr Scalar_ operator()(
      Scalar_ x, Scalar_ min, Scalar_ max, [[maybe_unused]] int dim) const {
    return internal::angle_distance_box(x, min, max);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return std::abs(x);
  }
};

//! \brief The metric_se2_squared measures distances in Euclidean space between
//! Euclidean motions.
//! \details Named after the Special Euclidean group of dimension 2.
//! For more details:
//! * https://en.wikipedia.org/wiki/Euclidean_group
struct metric_se2_squared {
  //! \brief This tag specifies the supported space by this metric.
  using space_category = topological_space_tag;

  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename InputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1, InputSentinel1_, InputIterator2_ begin2) const {
    return internal::sum(
               begin1, begin1 + 2, begin2, internal::squared_distance_fn()) +
           internal::squared_angle_distance(*(begin1 + 2), *(begin2 + 2));
  }

  //! \brief Calculates the squared distance between coordinate \p x and the box
  //! defined by [ \p min, \p max ].
  template <typename Scalar_>
  constexpr Scalar_ operator()(
      Scalar_ x, Scalar_ min, Scalar_ max, int dim) const {
    if (dim < 2) {
      return internal::squared_distance_box(x, min, max);
    } else {
      return internal::squared_angle_distance_box(x, min, max);
    }
  }

  //! \brief Returns the squared value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ x) const {
    return internal::squared(x);
  }
};

}  // namespace pico_tree
