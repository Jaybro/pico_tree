#pragma once

#include "point.hpp"

//! \brief Example point set adaptor that shows which functions need to be
//! implemented.
template <typename Index_, typename Point_>
class PicoAdaptor {
 public:
  using Index = Index_;
  using Point = Point_;
  using Scalar = typename Point::Scalar;
  static constexpr int Dim = Point::Dim;

  explicit PicoAdaptor(std::vector<Point> const& points) : points_(points) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, int const dim) const {
    return points_[idx](dim);
  }

  //! Returns dimension \p dim of point \p point.
  inline Scalar operator()(Point const& point, int const dim) const {
    return point(dim);
  }

  //! Returns the amount of spatial dimensions of the points.
  inline int num_dimensions() const { return Dim; };

  //! Returns the number of points.
  inline Index num_points() const {
    return static_cast<Index>(points_.size());
  };

 private:
  std::vector<Point> const& points_;
};
