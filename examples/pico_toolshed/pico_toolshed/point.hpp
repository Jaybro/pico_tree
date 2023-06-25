#pragma once

#include <iostream>
#include <pico_tree/internal/point.hpp>
#include <pico_tree/point_traits.hpp>
#include <random>
#include <vector>

template <typename Scalar_, std::size_t Dim_>
struct Point : public pico_tree::internal::Point<Scalar_, Dim_> {
  // Dynamic capability disabled.
  static_assert(
      Dim_ != pico_tree::kDynamicSize && Dim_ > 0,
      "DIM_MUST_NOT_BE_DYNAMIC_AND_>_0");

  using pico_tree::internal::Point<Scalar_, Dim_>::elems_;
  using pico_tree::internal::Point<Scalar_, Dim_>::size;
  using typename pico_tree::internal::Point<Scalar_, Dim_>::ScalarType;
  using typename pico_tree::internal::Point<Scalar_, Dim_>::SizeType;

  inline Point& operator+=(ScalarType const v) {
    for (SizeType i = 0; i < size(); ++i) {
      elems_[i] += v;
    }
    return *this;
  }

  inline Point operator+(ScalarType const v) const {
    Point p = *this;
    p += v;
    return p;
  }

  inline Point& operator-=(ScalarType const v) {
    for (SizeType i = 0; i < size(); ++i) {
      elems_[i] -= v;
    }
    return *this;
  }

  inline Point operator-(ScalarType const v) const {
    Point p = *this;
    p -= v;
    return p;
  }

  inline void Fill(ScalarType const v) {
    for (SizeType i = 0; i < size(); ++i) {
      elems_[i] = v;
    }
  }

  template <typename OtherScalarType>
  inline Point<OtherScalarType, Dim_> Cast() const {
    Point<OtherScalarType, Dim_> other;
    for (SizeType i = 0; i < size(); ++i) {
      other.elems_[i] = static_cast<OtherScalarType>(elems_[i]);
    }
    return other;
  }
};

namespace pico_tree {

template <typename Scalar_, std::size_t Dim_>
struct PointTraits<Point<Scalar_, Dim_>> {
  using PointType = Point<Scalar_, Dim_>;
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  inline static ScalarType const* Coords(PointType const& point) {
    return point.data();
  }

  inline static std::size_t constexpr Sdim(PointType const& point) {
    return point.size();
  }
};

}  // namespace pico_tree

template <typename Scalar_, std::size_t Dim_>
inline std::ostream& operator<<(
    std::ostream& s, Point<Scalar_, Dim_> const& p) {
  s << p[0];
  for (std::size_t i = 1; i < Dim_; ++i) {
    s << " " << p[i];
  }
  return s;
}

using Point1f = Point<float, 1>;
using Point2f = Point<float, 2>;
using Point3f = Point<float, 3>;
using Point1d = Point<double, 1>;
using Point2d = Point<double, 2>;
using Point3d = Point<double, 3>;

// Generates n random points uniformly distributed between the box defined by
// min and max.
template <typename Point>
inline std::vector<Point> GenerateRandomN(
    std::size_t n,
    typename Point::ScalarType min,
    typename Point::ScalarType max) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point::ScalarType> dist(min, max);

  std::vector<Point> random(n);
  for (auto& p : random) {
    for (std::size_t i = 0; i < Point::Dim; ++i) {
      p[i] = dist(e2);
    }
  }

  return random;
}

// Generates n random points uniformly distributed over a box of size size.
template <typename Point>
inline std::vector<Point> GenerateRandomN(
    std::size_t n, typename Point::ScalarType size) {
  return GenerateRandomN<Point>(n, typename Point::ScalarType(0.0), size);
}
