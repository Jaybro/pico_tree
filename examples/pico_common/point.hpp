#pragma once

#include <array>
#include <random>
#include <vector>

//! Demo point type.
template <typename Scalar_, int Dims_>
class Point {
 public:
  using Scalar = Scalar_;
  static constexpr int Dims = Dims_;

  inline Scalar const& operator()(int dim) const { return data[dim]; }
  inline Scalar& operator()(int dim) { return data[dim]; }

  inline void Fill(Scalar const v) {
    for (int i = 0; i < Dims; ++i) {
      data[i] = v;
    }
  }

  Scalar data[Dims];
};

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
  std::uniform_real_distribution<> dist(0, size);

  Point p;
  for (int i = 0; i < Point::Dims; ++i) {
    p(i) = dist(e2);
  }

  return p;
}

//! Generates \p n points in a square of size \p size .
template <typename Point>
std::vector<Point> GenerateRandomN(int n, typename Point::Scalar size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, size);

  std::vector<Point> random(n);
  for (auto& p : random) {
    for (int i = 0; i < Point::Dims; ++i) {
      p(i) = dist(e2);
    }
  }

  return random;
}
