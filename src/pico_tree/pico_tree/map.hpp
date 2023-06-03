#pragma once

#include <type_traits>

#include "core.hpp"
#include "point_traits.hpp"

namespace pico_tree {

namespace internal {

template <typename Element_, Size Size_>
struct MapStorage {
  constexpr MapStorage(Element_* data, Size) noexcept : data(data) {}

  Element_* data;
  static Size constexpr size = Size_;
};

template <typename Element_>
struct MapStorage<Element_, kDynamicSize> {
  constexpr MapStorage(Element_* data, Size size) noexcept
      : data(data), size(size) {}

  Element_* data;
  Size size;
};

template <typename Element_, Size Size_>
class Map {
 public:
  static_assert(std::is_object_v<Element_>, "ELEMENT_NOT_AN_OBJECT_TYPE");
  static_assert(
      !std::is_abstract_v<Element_>, "ELEMENT_CANNOT_BE_AN_ABSTRACT_TYPE");
  static_assert(
      Size_ == kDynamicSize || Size_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ElementType = Element_;
  using ValueType = std::remove_cv_t<Element_>;
  using SizeType = Size;
  static SizeType constexpr SizeValue = Size_;

  constexpr Map(ElementType* data) noexcept : storage_(data, SizeValue) {}

  constexpr Map(ElementType* data, SizeType size) noexcept
      : storage_(data, size) {}

  template <typename ContiguousAccessIterator_>
  constexpr Map(
      ContiguousAccessIterator_ begin, ContiguousAccessIterator_ end) noexcept
      : storage_(&(*begin), static_cast<SizeType>(end - begin)) {}

  constexpr ElementType& operator[](SizeType i) const {
    return storage_.data[i];
  }

  constexpr ElementType* data() const noexcept { return storage_.data; }

  constexpr SizeType size() const noexcept { return storage_.size; }

 protected:
  MapStorage<Element_, Size_> storage_;
};

template <typename Scalar_, Size Dim_>
struct SpaceMapMatrixStorage {
  constexpr SpaceMapMatrixStorage(Scalar_* data, Size size, Size)
      : data(data), size(size) {}

  Scalar_* data;
  Size size;
  static Size constexpr sdim = Dim_;
};

template <typename Scalar_>
struct SpaceMapMatrixStorage<Scalar_, kDynamicSize> {
  constexpr SpaceMapMatrixStorage(Scalar_* data, Size size, Size sdim)
      : data(data), size(size), sdim(sdim) {}

  Scalar_* data;
  Size size;
  Size sdim;
};

}  // namespace internal

//! \brief The PointMap class provides a point interface for an array of
//! scalars.
template <typename Scalar_, Size Dim_>
class PointMap : protected internal::Map<Scalar_, Dim_> {
 private:
  using Base = internal::Map<Scalar_, Dim_>;

 public:
  static_assert(std::is_arithmetic_v<Scalar_>, "SCALAR_NOT_AN_ARITHMETIC_TYPE");

  using ScalarType = typename Base::ValueType;
  using CvScalarType = typename Base::ElementType;
  using typename internal::Map<Scalar_, Dim_>::SizeType;
  static SizeType constexpr Dim = Base::SizeValue;

  using internal::Map<Scalar_, Dim_>::Map;
  using internal::Map<Scalar_, Dim_>::operator[];
  using internal::Map<Scalar_, Dim_>::data;
  using internal::Map<Scalar_, Dim_>::size;
};

//! \brief The SpaceMap class provides a space interface for an array of points.
template <typename Point_>
class SpaceMap : protected internal::Map<Point_, kDynamicSize> {
  using Base = internal::Map<Point_, kDynamicSize>;

 public:
  using PointType = typename Base::ValueType;
  using CvPointType = typename Base::ElementType;
  using ScalarType = typename PointTraits<PointType>::ScalarType;
  using SizeType = Size;
  static SizeType constexpr Dim = PointTraits<PointType>::Dim;

  static_assert(
      Dim != kDynamicSize, "SPACE_MAP_OF_POINT_DOES_NOT_SUPPORT_DYNAMIC_DIM");

  using internal::Map<Point_, kDynamicSize>::Map;
  using internal::Map<Point_, kDynamicSize>::operator[];
  using internal::Map<Point_, kDynamicSize>::data;
  using internal::Map<Point_, kDynamicSize>::size;

  constexpr SizeType sdim() const { return Dim; }
};

//! \brief The SpaceMap class provides a space interface for an array of
//! scalars.
template <typename Scalar_, Size Dim_>
class SpaceMap<PointMap<Scalar_, Dim_>> {
 public:
  using PointType = PointMap<Scalar_, Dim_>;
  using ScalarType = typename PointType::ScalarType;
  using CvScalarType = typename PointType::CvScalarType;
  using SizeType = typename PointType::SizeType;
  static SizeType constexpr Dim = PointType::Dim;

  constexpr SpaceMap(CvScalarType* data, SizeType size) noexcept
      : storage_(data, size, Dim) {}

  constexpr SpaceMap(CvScalarType* data, SizeType size, SizeType sdim) noexcept
      : storage_(data, size, sdim) {}

  constexpr PointType operator[](SizeType i) const noexcept {
    return {storage_.data + i * storage_.sdim, storage_.sdim};
  }

  constexpr CvScalarType* data() const noexcept { return storage_.data; }

  constexpr SizeType size() const noexcept { return storage_.size; }

  constexpr SizeType sdim() const noexcept { return storage_.sdim; }

 protected:
  internal::SpaceMapMatrixStorage<Scalar_, Dim_> storage_;
};

}  // namespace pico_tree
