#pragma once

#include <pybind11/numpy.h>

#include <pico_tree/std_traits.hpp>

#include "core.hpp"

namespace py = pybind11;

namespace pyco_tree {

template <typename Map_>
class Block {
 public:
  using ScalarType = typename Map_::ScalarType;
  static int constexpr Dim = Map_::Dim;

  inline Block(ScalarType const* const data, Map_ const& space)
      : data_(data), space_(space) {}

  inline ScalarType const& operator()(int const i) const { return data_[i]; }

  inline ScalarType const* data() const { return data_; }

  inline int sdim() const { return space_.sdim(); }

 private:
  ScalarType const* const data_;
  Map_ const& space_;
};

// TODO Probably remove the Index argument in the future.
template <typename Scalar, int Dim_, typename Index>
class Map {
 public:
  using ScalarType = Scalar;
  static int constexpr Dim = Dim_;
  using IndexType = Index;

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

  inline Block<Map> operator()(Index const idx) const {
    return Block<Map>(data_ + idx * sdim_, *this);
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

template <typename Map_>
struct StdPointTraits<typename pyco_tree::Block<Map_>> {
  using ScalarType = typename Map_::ScalarType;
  static constexpr int Dim = Map_::Dim;

  inline static ScalarType const* Coords(pyco_tree::Block<Map_> const& point) {
    return point.data();
  }

  inline static int constexpr Sdim(pyco_tree::Block<Map_> const& point) {
    return point.sdim();
  }
};

}  // namespace pico_tree

namespace pyco_tree {

template <typename Scalar_, int Dim_, typename Index_>
struct MapTraits {
  using SpaceType = Map<Scalar_, Dim_, Index_>;
  using PointType = Block<SpaceType>;
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
