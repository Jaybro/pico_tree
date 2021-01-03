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
template <typename Index, typename Point>
class PicoAdaptor {
 public:
  using IndexType = Index;
  using PointType = Point;
  using ScalarType = typename Point::ScalarType;
  static constexpr int Dim = Point::Dim;

  explicit PicoAdaptor(std::vector<Point> const& points) : points_(points) {}

  //! \brief Returns the point at index \p idx.
  inline Point const& operator()(Index const idx) const { return points_[idx]; }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return Dim; };

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(points_.size()); }

 private:
  //! \brief A reference to the actual point data.
  //! \details Using a reference avoids unwanted copies but it doesn't allow the
  //! KdTree to fully own everything. This means we always have to keep track of
  //! at least 2 variables:
  //!
  //! * std::vector<> points;
  //! * pico_tree::KdTree<> tree(PicoAdaptor<>(points), 10);
  //!
  //! We could allow the constructor of this class to std::move() the points
  //! into the points_ member if we removed the const&. This would allow the
  //! KdTree to fully own everything:
  //!
  //! * pico_tree::KdTree<> tree(PicoAdaptor<>(std::move(points)), 10);
  std::vector<Point> const& points_;
};
