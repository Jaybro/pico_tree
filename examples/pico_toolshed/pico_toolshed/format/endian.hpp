#pragma once

#include <cstddef>
#include <type_traits>

namespace pico_tree::internal {

template <typename T>
T swap_endian(T const& u) {
  T s;

  std::byte const* up = reinterpret_cast<std::byte const*>(&u);
  std::byte* sp = reinterpret_cast<std::byte*>(&s);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    sp[i] = up[sizeof(T) - i - 1];
  }

  return s;
}

template <typename T_>
T_ big_endian_to_native(T_ v) {
  static_assert(std::is_integral_v<T_>, "NOT_AN_INTEGRAL_TYPE");
  static_assert(sizeof(T_) <= sizeof(std::size_t), "SIZE_UNSUPPORTED");

  std::byte* v_object = reinterpret_cast<std::byte*>(&v);

  T_ native{};
  for (std::size_t i = 0; i < sizeof(T_); ++i) {
    native |= static_cast<T_>(
        static_cast<std::size_t>(v_object[i]) << ((sizeof(T_) - 1 - i) * 8));
  }

  return native;
}

//! \brief Stores the value of an integral type assuming big endian byte
//! ordering.
//! \details The idea is to be oblivious to the native endianness. When a big
//! endian value is read it can be converted to its native counterpart. If the
//! native endianness equals big endian then we're basically making a very
//! elaborate copy. However, use of this class should be quite minimal.
template <typename T_>
struct big_endian {
  static_assert(std::is_integral_v<T_>, "NOT_AN_INTEGRAL_TYPE");

  T_ operator()() const { return big_endian_to_native(value); }

  operator T_() const { return this->operator()(); }

  T_ value;
};

}  // namespace pico_tree::internal
