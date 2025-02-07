#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/map.hpp"

namespace pico_tree::internal {

template <typename Scalar_, size_t Dim_>
struct matrix_space_storage {
  constexpr matrix_space_storage(size_t size, size_t)
      : size(size), elems(size * sdim) {}

  size_t size;
  static constexpr size_t sdim = Dim_;
  std::vector<Scalar_> elems;
};

template <typename Scalar_>
struct matrix_space_storage<Scalar_, dynamic_size> {
  constexpr matrix_space_storage(size_t size, size_t sdim)
      : size(size), sdim(sdim), elems(size * sdim) {}

  size_t size;
  size_t sdim;
  std::vector<Scalar_> elems;
};

template <typename Scalar_, size_t Dim_>
class matrix_space {
 public:
  using scalar_type = Scalar_;
  using size_type = pico_tree::size_t;
  static size_type constexpr dim = Dim_;

  constexpr matrix_space(size_type size) noexcept : storage_(size, dim) {}

  constexpr matrix_space(size_type size, size_type sdim) noexcept
      : storage_(size, sdim) {}

  constexpr point_map<Scalar_ const, Dim_> operator[](
      size_type i) const noexcept {
    return {data(i), storage_.sdim};
  }

  constexpr point_map<Scalar_, Dim_> operator[](size_type i) noexcept {
    return {data(i), storage_.sdim};
  }

  constexpr scalar_type const* data() const noexcept {
    return storage_.elems.data();
  }

  constexpr scalar_type* data() noexcept { return storage_.elems.data(); }

  constexpr scalar_type const* data(size_type i) const noexcept {
    return storage_.elems.data() + i * storage_.sdim;
  }

  constexpr scalar_type* data(size_type i) noexcept {
    return storage_.elems.data() + i * storage_.sdim;
  }

  constexpr size_type size() const noexcept { return storage_.size; }

  constexpr size_type sdim() const noexcept { return storage_.sdim; }

 protected:
  internal::matrix_space_storage<Scalar_, Dim_> storage_;
};

}  // namespace pico_tree::internal
