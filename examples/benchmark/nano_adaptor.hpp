#pragma once

#include <nanoflann.hpp>
#include <pico_toolshed/point.hpp>

//! Demo point set adaptor for a vector of points.
template <typename Index, typename Point>
class NanoAdaptor {
 private:
  using Scalar = typename Point::ScalarType;

 public:
  using IndexType = Index;
  using PointType = Point;
  using ScalarType = typename Point::ScalarType;
  static constexpr int Dim = Point::Dim;

  NanoAdaptor(std::vector<Point> const& points) : points_(points) {}

  //! \brief Returns the number of points.
  inline Index kdtree_get_point_count() const {
    return static_cast<Index>(points_.size());
  }

  //! \brief Returns the dim'th component of the idx'th point in the class:
  inline Scalar kdtree_get_pt(Index const idx, Index const dim) const {
    return points_[idx].data[dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const {
    return false;
  }

 private:
  std::vector<Point> const& points_;
};
