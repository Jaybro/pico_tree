#pragma once

#include <pico_tree/kd_tree.hpp>
#include <pico_tree/range_tree.hpp>

#include "point.hpp"

//! Demo point set adapter for a vector of points.
template <typename Index_, typename Point_>
class PicoPointSet {
 public:
  using Index = Index_;
  using Point = Point_;
  using Scalar = typename Point::Scalar;
  static constexpr int Dims = Point::Dims;

  PicoPointSet(std::vector<Point> const& points) : points_(points) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, Index const dim) const {
    return points_[idx](dim);
  }

  //! Returns dimension \p dim of point \p point.
  inline Scalar operator()(Point const& point, Index const dim) const {
    return point(dim);
  }

  //! Returns the amount of spatial dimensions of the points.
  inline Index num_dimensions() const { return Dims; };

  //! Returns the number of points.
  inline Index num_points() const { return points_.size(); };

 private:
  std::vector<Point> const& points_;
};

template <typename PointSet>
using KdTree = pico_tree::KdTree<
    typename PointSet::Index,
    typename PointSet::Scalar,
    PointSet::Dims,
    PointSet>;

template <typename PointSet>
using KdTreeRt = pico_tree::KdTree<
    typename PointSet::Index,
    typename PointSet::Scalar,
    pico_tree::kRuntimeDims,
    PointSet>;

template <typename PointSet>
using RangeTree2d = pico_tree::
    RangeTree2d<typename PointSet::Index, typename PointSet::Scalar, PointSet>;

using PicoPointSet1d = PicoPointSet<Index, Point1d>;
using PicoPointSet2d = PicoPointSet<Index, Point2d>;
using PicoPointSet3d = PicoPointSet<Index, Point3d>;
