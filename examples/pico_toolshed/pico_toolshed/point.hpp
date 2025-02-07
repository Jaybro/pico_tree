#pragma once

#include <iostream>
#include <pico_tree/internal/point.hpp>
#include <pico_tree/point_traits.hpp>
#include <random>
#include <vector>

namespace pico_tree {

template <typename Scalar_, std::size_t Dim_>
struct point : public pico_tree::internal::point<Scalar_, Dim_> {
  // Dynamic capability disabled.
  static_assert(
      Dim_ != pico_tree::dynamic_size && Dim_ > 0,
      "DIM_MUST_NOT_BE_DYNAMIC_AND_>_0");

  using pico_tree::internal::point<Scalar_, Dim_>::elems_;
  using pico_tree::internal::point<Scalar_, Dim_>::size;
  using pico_tree::internal::point<Scalar_, Dim_>::fill;
  using pico_tree::internal::point<Scalar_, Dim_>::normalize;
  using typename pico_tree::internal::point<Scalar_, Dim_>::scalar_type;
  using typename pico_tree::internal::point<Scalar_, Dim_>::size_type;

  inline point& operator+=(scalar_type const v) {
    for (size_type i = 0; i < size(); ++i) {
      elems_[i] += v;
    }
    return *this;
  }

  inline point operator+(scalar_type const v) const {
    point p = *this;
    p += v;
    return p;
  }

  inline point& operator-=(scalar_type const v) {
    for (size_type i = 0; i < size(); ++i) {
      elems_[i] -= v;
    }
    return *this;
  }

  inline point operator-(scalar_type const v) const {
    point p = *this;
    p -= v;
    return p;
  }

  template <typename OtherScalarType_>
  inline point<OtherScalarType_, Dim_> cast() const {
    point<OtherScalarType_, Dim_> other;
    for (size_type i = 0; i < size(); ++i) {
      other.elems_[i] = static_cast<OtherScalarType_>(elems_[i]);
    }
    return other;
  }
};

template <typename Scalar_, std::size_t Dim_>
struct point_traits<point<Scalar_, Dim_>> {
  using point_type = point<Scalar_, Dim_>;
  using scalar_type = Scalar_;
  static std::size_t constexpr dim = Dim_;

  inline static scalar_type const* data(point_type const& p) {
    return p.data();
  }

  inline static std::size_t constexpr size(point_type const& p) {
    return p.size();
  }
};

template <typename Scalar_, std::size_t Dim_>
inline std::ostream& operator<<(
    std::ostream& s, point<Scalar_, Dim_> const& p) {
  s << p[0];
  for (std::size_t i = 1; i < Dim_; ++i) {
    s << " " << p[i];
  }
  return s;
}

using point_1f = point<float, 1>;
using point_2f = point<float, 2>;
using point_3f = point<float, 3>;
using point_1d = point<double, 1>;
using point_2d = point<double, 2>;
using point_3d = point<double, 3>;

// Generates n random points uniformly distributed between the box defined by
// min and max.
template <typename Point_>
inline std::vector<Point_> generate_random_n(
    std::size_t n,
    typename Point_::scalar_type min,
    typename Point_::scalar_type max) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point_::scalar_type> dist(min, max);

  std::vector<Point_> random(n);
  for (auto& p : random) {
    for (std::size_t i = 0; i < Point_::dim; ++i) {
      p[i] = dist(e2);
    }
  }

  return random;
}

// Generates n random points uniformly distributed over a box of size size.
template <typename Point_>
inline std::vector<Point_> generate_random_n(
    std::size_t n, typename Point_::scalar_type size) {
  return generate_random_n<Point_>(n, typename Point_::scalar_type(0.0), size);
}

}  // namespace pico_tree
