#pragma once

#include <nanoflann.hpp>
#include <point.hpp>

//! Demo point set adaptor for a vector of points.
template <typename Index_, typename Point_>
class NanoAdaptor {
 public:
  using Index = Index_;
  using Point = Point_;
  using Scalar = typename Point::Scalar;
  static constexpr int Dims = Point::Dim;

  NanoAdaptor(std::vector<Point> const& points) : points_(points) {}

  //! Returns the number of points.
  inline Index kdtree_get_point_count() const { return points_.size(); }

  //! Returns the dim'th component of the idx'th point in the class:
  inline Scalar kdtree_get_pt(Index const idx, Index const dim) const {
    return points_[idx](dim);
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
