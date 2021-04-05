#pragma once

#include <iostream>
#include <pico_tree/std_traits.hpp>
#include <random>
#include <vector>

//! \brief Example point type.
//! \tparam Scalar_ Coordinate value type.
//! \tparam Dim_ The dimension of the space in which the point resides.
template <typename Scalar_, int Dim_>
class Point {
 public:
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline ScalarType const& operator()(int const i) const { return data[i]; }
  inline ScalarType& operator()(int const i) { return data[i]; }

  inline void Fill(ScalarType const v) {
    for (int i = 0; i < Dim; ++i) {
      data[i] = v;
    }
  }

  ScalarType data[Dim];
};

namespace pico_tree {

//! \brief Example point traits implementation for use with
//! pico_tree::StdTraits.
//! \details An implementation of StdPointTraits<PointType> *must* provide all
//! the details of this example implementation.
template <typename Scalar_, int Dim_>
struct StdPointTraits<Point<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline static ScalarType const* Coords(Point<ScalarType, Dim> const& point) {
    return point.data;
  }

  inline static int constexpr Sdim(Point<ScalarType, Dim> const&) {
    return Dim_;
  }
};

}  // namespace pico_tree

template <typename Scalar_, int Dim_>
inline std::ostream& operator<<(
    std::ostream& s, Point<Scalar_, Dim_> const& p) {
  s << p(0);
  for (int i = 1; i < Dim_; ++i) {
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

//! Generates \p n points in a square of size \p size .
template <typename Point>
std::vector<Point> GenerateRandomN(int n, typename Point::ScalarType size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point::ScalarType> dist(0, size);

  std::vector<Point> random(n);
  for (auto& p : random) {
    for (int i = 0; i < Point::Dim; ++i) {
      p(i) = dist(e2);
    }
  }

  return random;
}
