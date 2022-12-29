#pragma once

#include <iostream>
#include <pico_tree/std_traits.hpp>
#include <random>
#include <vector>

// Example point type.
template <typename Scalar_, std::size_t Dim_>
class Point {
  static_assert(Dim_ > 0, "INVALID_SPATIAL_DIMENSION_POINT");

 public:
  using ScalarType = Scalar_;
  using SizeType = std::size_t;
  static SizeType constexpr Dim = Dim_;

  inline ScalarType const& operator()(SizeType const i) const {
    return data[i];
  }

  inline ScalarType& operator()(SizeType const i) { return data[i]; }

  inline Point& operator+=(ScalarType const v) {
    for (SizeType i = 0; i < Dim; ++i) {
      data[i] += v;
    }
    return *this;
  }

  inline Point operator+(ScalarType const v) const {
    Point p = *this;
    p += v;
    return p;
  }

  inline Point& operator-=(ScalarType const v) {
    for (SizeType i = 0; i < Dim; ++i) {
      data[i] -= v;
    }
    return *this;
  }

  inline Point operator-(ScalarType const v) const {
    Point p = *this;
    p -= v;
    return p;
  }

  inline void Fill(ScalarType const v) {
    for (SizeType i = 0; i < Dim; ++i) {
      data[i] = v;
    }
  }

  template <typename OtherScalarType>
  inline Point<OtherScalarType, Dim> Cast() const {
    Point<OtherScalarType, Dim> other;
    for (SizeType i = 0; i < Dim; ++i) {
      other.data[i] = static_cast<OtherScalarType>(data[i]);
    }
    return other;
  }

  ScalarType data[Dim];
};

// A specialization of StdPointTraits must be defined within the pico_tree
// namespace and provide all the details of this example.
namespace pico_tree {

template <typename Scalar_, std::size_t Dim_>
struct StdPointTraits<Point<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static std::size_t constexpr Dim = Dim_;

  // Returns a pointer to the coordinates of the input point.
  inline static ScalarType const* Coords(Point<ScalarType, Dim_> const& point) {
    return point.data;
  }

  // Returns the spatial dimension of the input point. Note that the input
  // argument is ignored because the spatial dimension is known at compile time.
  inline static std::size_t constexpr Sdim(Point<ScalarType, Dim_> const&) {
    return Dim_;
  }
};

}  // namespace pico_tree

template <typename Scalar_, std::size_t Dim_>
inline std::ostream& operator<<(
    std::ostream& s, Point<Scalar_, Dim_> const& p) {
  s << p(0);
  for (std::size_t i = 1; i < Dim_; ++i) {
    s << " " << p(i);
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
      p(i) = dist(e2);
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
