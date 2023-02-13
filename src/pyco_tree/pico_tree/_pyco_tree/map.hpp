#pragma once

#include <pico_tree/core.hpp>
#include <type_traits>

namespace pico_tree {

namespace internal {

template <typename Element_, Size Size_>
struct MapStorage {
  constexpr MapStorage(Element_* data, Size) noexcept : data(data) {}

  Element_* data;
  static Size constexpr size = Size_;
};

template <typename Element_>
struct MapStorage<Element_, kDynamicDim> {
  constexpr MapStorage(Element_* data, Size size) noexcept
      : data(data), size(size) {}

  Element_* data;
  Size size;
};

template <typename Element_, Size Size_>
class Map {
 public:
  static_assert(std::is_object<Element_>::value, "ELEMENT_NOT_AN_OBJECT_TYPE");
  static_assert(
      !std::is_abstract<Element_>::value, "ELEMENT_CANNOT_BE_AN_ABSTRACT_TYPE");
  static_assert(
      Size_ == kDynamicDim || Size_ > 0, "DIM_MUST_BE_DYNAMIC_OR_>_0");

  using ElementType = Element_;
  using ValueType = typename std::remove_cv<Element_>::type;
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
struct SpaceMapMatrixStorage<Scalar_, kDynamicDim> {
  constexpr SpaceMapMatrixStorage(Scalar_* data, Size size, Size sdim)
      : data(data), size(size), sdim(sdim) {}

  Scalar_* data;
  Size size;
  Size sdim;
};

}  // namespace internal

template <typename Point_>
struct StdPointTraits;

//! \brief The PointMap class provides an interface for accessing a raw pointer
//! as a Point, allowing easy access to its coordinates.
template <typename Scalar_, Size Dim_>
class PointMap : protected internal::Map<Scalar_, Dim_> {
 private:
  using Base = internal::Map<Scalar_, Dim_>;

 public:
  static_assert(
      std::is_arithmetic<Scalar_>::value, "SCALAR_NOT_AN_ARITHMETIC_TYPE");

  using ScalarType = typename Base::ValueType;
  using CvScalarType = typename Base::ElementType;
  using typename internal::Map<Scalar_, Dim_>::SizeType;
  static SizeType constexpr Dim = Base::SizeValue;

  using internal::Map<Scalar_, Dim_>::Map;
  using internal::Map<Scalar_, Dim_>::operator[];
  using internal::Map<Scalar_, Dim_>::data;
  using internal::Map<Scalar_, Dim_>::size;
};

//! \brief The SpaceMap class provides an interface for accessing a raw pointer
//! as a Space, allowing easy access to its points via a PointMap interface.
template <typename Point_>
class SpaceMap : protected internal::Map<Point_, kDynamicDim> {
 private:
  using Base = internal::Map<Point_, kDynamicDim>;

 public:
  using PointType = typename Base::ValueType;
  using CvPointType = typename Base::ElementType;
  using ScalarType = typename StdPointTraits<PointType>::ScalarType;
  using SizeType = Size;
  static SizeType constexpr Dim = StdPointTraits<PointType>::Dim;

  using internal::Map<Point_, kDynamicDim>::Map;
  using internal::Map<Point_, kDynamicDim>::operator[];
  using internal::Map<Point_, kDynamicDim>::data;
  using internal::Map<Point_, kDynamicDim>::size;

  constexpr SizeType sdim() const { return Dim; }
};

//! \brief The SpaceMap class provides an interface for accessing a raw pointer
//! as a Space, allowing easy access to its points via a PointMap interface.
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
