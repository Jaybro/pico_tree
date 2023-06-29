#pragma once

#include <pico_tree/metric.hpp>

namespace pico_tree {

//! \brief L2 metric for measuring Euclidean distances between points.
//! \details https://en.wikipedia.org/wiki/Euclidean_distance
//! \see L1
class L2 {
 public:
  template <
      typename InputIterator1,
      typename InputSentinel1,
      typename OutputIterator2>
  constexpr auto operator()(
      InputIterator1 begin1,
      InputSentinel1 end1,
      OutputIterator2 begin2) const {
    return std::sqrt(
        internal::Sum(begin1, end1, begin2, internal::SquaredDistanceFn()));
  }

  //! \brief Calculates the distance between two coordinates.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x, Scalar_ const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  template <typename Scalar_>
  constexpr Scalar_ operator()(Scalar_ const x) const {
    return std::abs(x);
  }
};

}  // namespace pico_tree
