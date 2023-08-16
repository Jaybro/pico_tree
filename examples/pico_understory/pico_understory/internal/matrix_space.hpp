#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/map.hpp"

namespace pico_tree::internal {

template <typename Scalar_, Size Dim_>
struct MatrixSpaceStorage {
  constexpr MatrixSpaceStorage(Size size, Size)
      : size(size), elems(size * sdim) {}

  Size size;
  static Size constexpr sdim = Dim_;
  std::vector<Scalar_> elems;
};

template <typename Scalar_>
struct MatrixSpaceStorage<Scalar_, kDynamicSize> {
  constexpr MatrixSpaceStorage(Size size, Size sdim)
      : size(size), sdim(sdim), elems(size * sdim) {}

  Size size;
  Size sdim;
  std::vector<Scalar_> elems;
};

template <typename Scalar_, Size Dim_>
class MatrixSpace {
 public:
  using ScalarType = Scalar_;
  using SizeType = pico_tree::Size;
  static SizeType constexpr Dim = Dim_;

  constexpr MatrixSpace(SizeType size) noexcept : storage_(size, Dim) {}

  constexpr MatrixSpace(SizeType size, SizeType sdim) noexcept
      : storage_(size, sdim) {}

  constexpr PointMap<Scalar_ const, Dim_> operator[](
      SizeType i) const noexcept {
    return {data(i), storage_.sdim};
  }

  constexpr PointMap<Scalar_, Dim_> operator[](SizeType i) noexcept {
    return {data(i), storage_.sdim};
  }

  constexpr ScalarType const* data() const noexcept {
    return storage_.elems.data();
  }

  constexpr ScalarType* data() noexcept { return storage_.elems.data(); }

  constexpr ScalarType const* data(SizeType i) const noexcept {
    return storage_.elems.data() + i * storage_.sdim;
  }

  constexpr ScalarType* data(SizeType i) noexcept {
    return storage_.elems.data() + i * storage_.sdim;
  }

  constexpr SizeType size() const noexcept { return storage_.size; }

  constexpr SizeType sdim() const noexcept { return storage_.sdim; }

 protected:
  internal::MatrixSpaceStorage<Scalar_, Dim_> storage_;
};

}  // namespace pico_tree::internal
