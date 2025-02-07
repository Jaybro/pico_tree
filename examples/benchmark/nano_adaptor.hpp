#pragma once

#include <nanoflann.hpp>
#include <pico_toolshed/point.hpp>

//! Demo point set adaptor for a vector of points.
template <typename Index_, typename Point_>
class nano_adaptor {
 public:
  using index_type = Index_;
  using point_type = Point_;
  using scalar_type = typename Point_::scalar_type;
  static constexpr int dim = static_cast<int>(Point_::dim);

  nano_adaptor(std::vector<Point_> const& points) : points_(points) {}

  //! \brief Returns the number of points.
  inline Index_ kdtree_get_point_count() const {
    return static_cast<Index_>(points_.size());
  }

  //! \brief Returns the dim'th component of the idx'th point in the class:
  inline scalar_type kdtree_get_pt(Index_ const idx, Index_ const dim) const {
    return points_[idx].data()[dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX_>
  bool kdtree_get_bbox(BBOX_& /*bb*/) const {
    return false;
  }

 private:
  std::vector<Point_> const& points_;
};
