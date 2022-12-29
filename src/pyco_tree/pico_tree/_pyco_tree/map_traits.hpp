#pragma once

#include <pico_tree/core.hpp>

namespace pico_tree {

template <typename Scalar_, Size Dim_>
class PointMap;

namespace internal {

template <typename Scalar_, Size Dim_>
class PointMapBase {
 public:
  static_assert(Dim_ == kDynamicDim || Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;

  inline ScalarType const& operator()(SizeType i) const { return data_[i]; }
  inline ScalarType& operator()(SizeType i) { return data_[i]; }

  inline ScalarType const* data() const { return data_; }
  inline ScalarType* data() { return data_; }

 protected:
  inline PointMapBase(ScalarType* data) : data_(data) {}
  inline ~PointMapBase() = default;

  ScalarType* data_;
};

template <typename Scalar_, Size Dim_>
class SpaceMapBase {
 public:
  static_assert(Dim_ == kDynamicDim || Dim_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;

  inline ScalarType const* data() const { return data_; }
  inline ScalarType* data() { return data_; }
  inline SizeType npts() const { return npts_; }

 protected:
  inline SpaceMapBase(ScalarType* data, SizeType npts)
      : data_(data), npts_(npts) {}

  ScalarType* data_;
  SizeType npts_;
};

}  // namespace internal

//! \brief The PointMap class provides an interface for accessing a raw pointer
//! as a Point, allowing easy access to its coordinates.
template <typename Scalar_, Size Dim_>
class PointMap : public internal::PointMapBase<Scalar_, Dim_> {
 public:
  using typename internal::PointMapBase<Scalar_, Dim_>::ScalarType;
  using typename internal::PointMapBase<Scalar_, Dim_>::SizeType;
  using internal::PointMapBase<Scalar_, Dim_>::Dim;

  inline PointMap(ScalarType* data)
      : internal::PointMapBase<ScalarType, Dim>(data) {}
  //! \private Streamlines interfaces with the kDynamicDim overload.
  inline PointMap(ScalarType* data, SizeType)
      : internal::PointMapBase<ScalarType, Dim>(data) {}

  inline SizeType constexpr sdim() const { return Dim; }
};

//! \brief The PointMap class provides an interface for accessing a raw pointer
//! as a Point, allowing easy access to its coordinates.
template <typename Scalar_>
class PointMap<Scalar_, kDynamicDim>
    : public internal::PointMapBase<Scalar_, kDynamicDim> {
 public:
  using typename internal::PointMapBase<Scalar_, kDynamicDim>::ScalarType;
  using typename internal::PointMapBase<Scalar_, kDynamicDim>::SizeType;
  using internal::PointMapBase<Scalar_, kDynamicDim>::Dim;

  inline PointMap(ScalarType* data, SizeType sdim)
      : internal::PointMapBase<ScalarType, Dim>(data), sdim_(sdim) {}

  inline SizeType sdim() const { return sdim_; }

 private:
  SizeType sdim_;
};

//! \brief The SpaceMap class provides an interface for accessing a raw pointer
//! as a Space, allowing easy access to its points via a PointMap interface.
template <typename Scalar_, Size Dim_>
class SpaceMap : public internal::SpaceMapBase<Scalar_, Dim_> {
 public:
  using typename internal::SpaceMapBase<Scalar_, Dim_>::ScalarType;
  using typename internal::SpaceMapBase<Scalar_, Dim_>::SizeType;
  using internal::SpaceMapBase<Scalar_, Dim_>::Dim;
  using internal::SpaceMapBase<Scalar_, Dim_>::data_;

  inline SpaceMap(ScalarType* data, SizeType npts)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts) {}
  //! \private Streamlines interfaces with the kDynamicDim overload.
  inline SpaceMap(ScalarType* data, SizeType npts, SizeType)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts) {}

  inline PointMap<ScalarType const, Dim> operator()(SizeType i) const {
    return {data_ + i * Dim};
  }
  inline PointMap<ScalarType, Dim> operator()(SizeType i) {
    return {data_ + i * Dim};
  }

  inline SizeType constexpr sdim() const { return Dim; }
};

//! \brief The SpaceMap class provides an interface for accessing a raw pointer
//! as a Space, allowing easy access to its points via a PointMap interface.
template <typename Scalar_>
class SpaceMap<Scalar_, kDynamicDim>
    : public internal::SpaceMapBase<Scalar_, kDynamicDim> {
 public:
  using typename internal::SpaceMapBase<Scalar_, kDynamicDim>::ScalarType;
  using typename internal::SpaceMapBase<Scalar_, kDynamicDim>::SizeType;
  using internal::SpaceMapBase<Scalar_, kDynamicDim>::Dim;
  using internal::SpaceMapBase<Scalar_, kDynamicDim>::data_;

  inline SpaceMap(ScalarType* data, SizeType npts, SizeType sdim)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts), sdim_(sdim) {}

  inline PointMap<ScalarType const, Dim> operator()(SizeType i) const {
    return {data_ + i * sdim_, sdim_};
  }
  inline PointMap<ScalarType, Dim> operator()(SizeType i) {
    return {data_ + i * sdim_, sdim_};
  }

  inline SizeType sdim() const { return sdim_; }

 private:
  SizeType sdim_;
};

template <typename Scalar_, Size Dim_>
struct StdPointTraits<PointMap<Scalar_, Dim_>> {
  using ScalarType = typename PointMap<Scalar_, Dim_>::ScalarType;
  using SizeType = typename PointMap<Scalar_, Dim_>::SizeType;
  static SizeType constexpr Dim = Dim_;

  inline static ScalarType const* Coords(PointMap<Scalar_, Dim_> const& point) {
    return point.data();
  }

  inline static SizeType Sdim(PointMap<Scalar_, Dim_> const& point) {
    return point.sdim();
  }
};

//! \brief MapTraits provides an interface for SpaceMap and points supported by
//! StdPointTraits.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Scalar_, Size Dim_, typename Index_ = int>
struct MapTraits {
  using SpaceType = SpaceMap<Scalar_, Dim_>;
  using PointType = PointMap<Scalar_ const, Dim_>;
  using ScalarType = Scalar_;
  using SizeType = Size;
  static SizeType constexpr Dim = Dim_;
  using IndexType = Index_;

  inline static SizeType SpaceSdim(SpaceType const& space) {
    return space.sdim();
  }

  inline static IndexType SpaceNpts(SpaceType const& space) {
    return static_cast<IndexType>(space.npts());
  }

  inline static PointType PointAt(SpaceType const& space, IndexType const idx) {
    return space(idx);
  }

  template <typename OtherPoint>
  inline static SizeType PointSdim(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Sdim(point);
  }

  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pico_tree
