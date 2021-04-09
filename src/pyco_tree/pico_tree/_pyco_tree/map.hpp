#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/std_traits.hpp>

#include "core.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename Index, typename Scalar, int Dim_>
class Map;

template <typename Index, typename Scalar, int Dim_>
class Block {
 public:
  inline Block(Scalar const* const data, Map<Index, Scalar, Dim_> const& space)
      : data_(data), space_(space) {}

  inline Scalar const& operator()(int const i) const { return data_[i]; }

  inline Scalar const* data() const { return data_; }

  inline int sdim() const { return space_.sdim(); }

 private:
  Scalar const* const data_;
  Map<Index, Scalar, Dim_> const& space_;
};

template <typename Index, typename Scalar, int Dim_>
class Map {
 public:
  using IndexType = Index;
  using ScalarType = Scalar;
  static constexpr int Dim = Dim_;

  explicit Map(py::array_t<Scalar, 0> const pts) {
    ArrayLayout<Scalar> layout(pts);

    if (layout.info.ndim != 2) {
      throw std::runtime_error("Array: ndim not 2.");
    }

    // We always want the memory layout to look like x,y,z,x,y,z,...,x,y,z.
    // This means that the shape of the inner dimension should equal the spatial
    // dimension of the KdTree.
    if (!IsDimCompatible<Dim>(
            static_cast<int>(layout.info.shape[layout.index_inner]))) {
      throw std::runtime_error(
          "Array: Incompatible KdTree sdim and Array inner stride.");
    }

    ThrowIfNotContiguous(layout);

    data_ = static_cast<Scalar*>(layout.info.ptr);
    sdim_ = static_cast<int>(layout.info.shape[layout.index_inner]);
    npts_ = static_cast<Index>(layout.info.shape[layout.index_outer]);
    row_major_ = layout.row_major;
  }

  inline Block<Index, Scalar, Dim_> operator()(Index const idx) const {
    return Block<Index, Scalar, Dim_>(data_ + idx * sdim_, *this);
  }

  inline Scalar const* data() const { return data_; }

  inline int sdim() const { return sdim_; }

  inline Index npts() const { return npts_; }

  inline bool row_major() const { return row_major_; }

 private:
  Scalar const* data_;
  int sdim_;
  Index npts_;
  bool row_major_;
};

}  // namespace pyco_tree

namespace pico_tree {

template <typename Index_, typename Scalar_, int Dim_>
struct StdPointTraits<typename pyco_tree::Block<Index_, Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline static ScalarType const* Coords(
      pyco_tree::Block<Index_, Scalar_, Dim_> const& point) {
    return point.data();
  }

  inline static int constexpr Sdim(
      pyco_tree::Block<Index_, Scalar_, Dim_> const& point) {
    return point.sdim();
  }
};

}  // namespace pico_tree

namespace pyco_tree {

template <typename Index_, typename Scalar_, int Dim_>
struct MapTraits {
  using SpaceType = Map<Index_, Scalar_, Dim_>;
  using PointType = Block<Index_, Scalar_, Dim_>;
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;
  using IndexType = Index_;

  inline static int SpaceSdim(SpaceType const& space) { return space.sdim(); }

  inline static IndexType SpaceNpts(SpaceType const& space) {
    return space.npts();
  }

  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space(idx);
  }

  template <typename OtherPoint>
  inline static int PointSdim(OtherPoint const& point) {
    return pico_tree::StdPointTraits<OtherPoint>::Sdim(point);
  }

  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return pico_tree::StdPointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pyco_tree
