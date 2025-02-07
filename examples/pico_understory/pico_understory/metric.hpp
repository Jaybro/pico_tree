#pragma once

#include <pico_tree/metric.hpp>

namespace pico_tree {

//! \brief metric_l2 metric for measuring Euclidean distances between points.
//! \details https://en.wikipedia.org/wiki/Euclidean_distance
//! \see metric_l1
class metric_l2 {
 public:
  template <
      typename InputIterator1_,
      typename InputSentinel1_,
      typename OutputIterator2_>
  constexpr auto operator()(
      InputIterator1_ begin1,
      InputSentinel1_ end1,
      OutputIterator2_ begin2) const {
    return std::sqrt(
        internal::sum(begin1, end1, begin2, internal::squared_distance_fn()));
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
