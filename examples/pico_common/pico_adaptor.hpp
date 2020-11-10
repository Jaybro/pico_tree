#pragma once

#include "point.hpp"

//! \brief Example point set. In this case the set is implemented as an adaptor
//! that wraps a vector of Point.
//! \details The following methods need to be implemented:
//! \code{.cpp}
//! inline Point const& operator()(Index const idx) const;
//! inline int sdim() const;
//! inline Index npts() const;
//! \endcode
template <typename Index_, typename Point_>
class PicoAdaptor {
 public:
  using Index = Index_;
  using Point = Point_;
  using Scalar = typename Point::Scalar;
  static constexpr int Dim = Point::Dim;

  explicit PicoAdaptor(std::vector<Point> const& points) : points_(points) {}

  //! \brief Returns the point at index \p idx.
  inline Point const& operator()(Index const idx) const { return points_[idx]; }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return Dim; };

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(points_.size()); };

 private:
  std::vector<Point> const& points_;
};
