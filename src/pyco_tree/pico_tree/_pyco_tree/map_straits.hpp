#pragma once

#include <pico_tree/core.hpp>

namespace pico_tree {

template <typename Scalar_, int Dim_>
class PointMap;

namespace internal {

template <typename Scalar_, int Dim_>
class PointMapBase {
 public:
  static_assert(
      Dim_ == kDynamicDim || Dim_ > 0,
      "SPATIAL_DIMENSION_MUST_DYNAMIC_OR_LARGER_THAN_0");

  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline ScalarType const& operator()(std::size_t i) const { return data_[i]; }
  inline ScalarType& operator()(std::size_t i) { return data_[i]; }

  inline ScalarType const* data() const { return data_; }
  inline ScalarType* data() { return data_; }

 protected:
  inline PointMapBase(ScalarType* data) : data_(data) {}
  inline ~PointMapBase() = default;

  ScalarType* data_;
};

template <typename Scalar_, int Dim_>
class SpaceMapBase {
 public:
  static_assert(
      Dim_ == kDynamicDim || Dim_ > 0,
      "SPATIAL_DIMENSION_MUST_DYNAMIC_OR_LARGER_THAN_0");

  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline ScalarType const* data() const { return data_; }
  inline ScalarType* data() { return data_; }
  inline std::size_t npts() const { return npts_; }

 protected:
  inline SpaceMapBase(ScalarType* data, std::size_t npts)
      : data_(data), npts_(npts) {}

  ScalarType* data_;
  std::size_t npts_;
};

}  // namespace internal

template <typename Scalar_, int Dim_>
class PointMap final : public internal::PointMapBase<Scalar_, Dim_> {
 public:
  using typename internal::PointMapBase<Scalar_, Dim_>::ScalarType;
  using internal::PointMapBase<Scalar_, Dim_>::Dim;

  inline PointMap(ScalarType* data)
      : internal::PointMapBase<ScalarType, Dim>(data) {}
  //! \private Streamlines interfaces with the kDynamicDim overload.
  inline PointMap(ScalarType* data, std::size_t)
      : internal::PointMapBase<ScalarType, Dim>(data) {}

  inline std::size_t constexpr sdim() const {
    return static_cast<std::size_t>(Dim);
  }
};

template <typename Scalar_>
class PointMap<Scalar_, kDynamicDim> final
    : public internal::PointMapBase<Scalar_, kDynamicDim> {
 public:
  using typename internal::PointMapBase<Scalar_, kDynamicDim>::ScalarType;
  using internal::PointMapBase<Scalar_, kDynamicDim>::Dim;

  inline PointMap(ScalarType* data, std::size_t sdim)
      : internal::PointMapBase<ScalarType, Dim>(data), sdim_(sdim) {}

  inline std::size_t sdim() const { return sdim_; }

 private:
  std::size_t sdim_;
};

template <typename Scalar_, int Dim_>
class SpaceMap final : public internal::SpaceMapBase<Scalar_, Dim_> {
 public:
  using typename internal::SpaceMapBase<Scalar_, Dim_>::ScalarType;
  using internal::SpaceMapBase<Scalar_, Dim_>::Dim;
  using internal::SpaceMapBase<Scalar_, Dim_>::data_;

  inline SpaceMap(ScalarType* data, std::size_t npts)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts) {}
  //! \private Streamlines interfaces with the kDynamicDim overload.
  inline SpaceMap(ScalarType* data, std::size_t npts, std::size_t)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts) {}

  inline PointMap<ScalarType const, Dim> operator()(std::size_t i) const {
    return {data_ + i * Dim};
  }
  inline PointMap<ScalarType, Dim> operator()(std::size_t i) {
    return {data_ + i * Dim};
  }

  inline std::size_t constexpr sdim() const {
    return static_cast<std::size_t>(Dim);
  }
};

template <typename Scalar_>
class SpaceMap<Scalar_, kDynamicDim> final
    : public internal::SpaceMapBase<Scalar_, kDynamicDim> {
 public:
  using typename internal::SpaceMapBase<Scalar_, kDynamicDim>::ScalarType;
  using internal::SpaceMapBase<Scalar_, kDynamicDim>::Dim;
  using internal::SpaceMapBase<Scalar_, kDynamicDim>::data_;

  inline SpaceMap(ScalarType* data, std::size_t npts, std::size_t sdim)
      : internal::SpaceMapBase<ScalarType, Dim>(data, npts), sdim_(sdim) {}

  inline PointMap<ScalarType const, Dim> operator()(std::size_t i) const {
    return {data_ + i * sdim_, sdim_};
  }
  inline PointMap<ScalarType, Dim> operator()(std::size_t i) {
    return {data_ + i * sdim_, sdim_};
  }

  inline std::size_t sdim() const { return sdim_; }

 private:
  std::size_t sdim_;
};

template <typename Scalar_, int Dim_>
struct StdPointTraits<PointMap<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static int constexpr Dim = Dim_;

  inline static ScalarType const* Coords(PointMap<Scalar_, Dim_> const& point) {
    return point.data();
  }

  inline static int Sdim(PointMap<Scalar_, Dim_> const& point) {
    return static_cast<int>(point.sdim());
  }
};

}  // namespace pico_tree
