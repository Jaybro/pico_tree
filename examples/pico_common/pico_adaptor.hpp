#pragma once

#include <pico_tree/kd_tree.hpp>
#include <pico_tree/range_tree.hpp>

#include "point.hpp"

//! Example point set adaptor that shows which functions need to be implemented.
template <typename Index_, typename Point_>
class PicoAdaptor {
 public:
  using Index = Index_;
  using Point = Point_;
  using Scalar = typename Point::Scalar;
  static constexpr int Dims = Point::Dims;

  explicit PicoAdaptor(std::vector<Point> const& points) : points_(points) {}

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

template <typename PicoAdaptor>
using KdTree = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using KdTreeRt = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    pico_tree::kRuntimeDims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using RangeTree2d = pico_tree::RangeTree2d<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor>;
