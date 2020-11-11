#pragma once

#include <array>
#include <iostream>
#include <random>
#include <vector>

//! \brief Example point type.
//! \details A point should at least implement the parenthesis operator:
//! \code{.cpp}
//! inline Scalar const& operator()(int i) const;
//! \endcode
//! \tparam Scalar_ Coordinate value type.
//! \tparam Dim_ The dimension of the space in which the point resides.
template <typename Scalar_, int Dim_>
class Point {
 public:
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;

  inline Scalar const& operator()(int const i) const { return data[i]; }
  inline Scalar& operator()(int const i) { return data[i]; }

  inline void Fill(Scalar const v) {
    for (int i = 0; i < Dim; ++i) {
      data[i] = v;
    }
  }

  Scalar data[Dim];
};

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

template <typename Point>
inline Point GenerateRandomP(typename Point::Scalar size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point::Scalar> dist(0, size);

  Point p;
  for (int i = 0; i < Point::Dim; ++i) {
    p(i) = dist(e2);
  }

  return p;
}

//! Generates \p n points in a square of size \p size .
template <typename Point>
std::vector<Point> GenerateRandomN(int n, typename Point::Scalar size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point::Scalar> dist(0, size);

  std::vector<Point> random(n);
  for (auto& p : random) {
    for (int i = 0; i < Point::Dim; ++i) {
      p(i) = dist(e2);
    }
  }

  return random;
}
