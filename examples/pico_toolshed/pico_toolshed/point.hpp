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
  static_assert(Dim_ > 0, "INVALID_SPATIAL_DIMENSION_POINT");

 public:
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline ScalarType const& operator()(int const i) const { return data[i]; }

  inline ScalarType& operator()(int const i) { return data[i]; }

  inline Point& operator+=(ScalarType const v) {
    for (int i = 0; i < Dim; ++i) {
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
    for (int i = 0; i < Dim; ++i) {
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
    for (int i = 0; i < Dim; ++i) {
      data[i] = v;
    }
  }

  template <typename OtherScalarType>
  inline Point<OtherScalarType, Dim> Cast() const {
    Point<OtherScalarType, Dim> other;
    for (int i = 0; i < Dim; ++i) {
      other.data[i] = static_cast<OtherScalarType>(data[i]);
    }
    return other;
  }

  ScalarType data[Dim];
};

namespace pico_tree {

//! \brief Example point traits implementation for use with
//! pico_tree::StdTraits.
//! \details An implementation of StdPointTraits<PointType> *must* provide all
//! the details of this example.
//!
//! A specialization of StdPointTraits must reside within the pico_tree
//! namespace.
template <typename Scalar_, int Dim_>
struct StdPointTraits<Point<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static ScalarType const* Coords(Point<ScalarType, Dim_> const& point) {
    return point.data;
  }

  //! \brief Returns the spatial dimension of a Point instance.
  //! \details The input argument is ignored because the spatial dimension is
  //! known at compile time. Also, the return value does not have to be
  //! constexpr in case the point type supports run-time dimensions.
  inline static int constexpr Sdim(Point<ScalarType, Dim_> const&) {
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

//! \brief Generates \p n random points uniformly distributed between the box
//! defined by \p min and \p max.
template <typename Point>
inline std::vector<Point> GenerateRandomN(
    int n, typename Point::ScalarType min, typename Point::ScalarType max) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Point::ScalarType> dist(min, max);

  std::vector<Point> random(n);
  for (auto& p : random) {
    for (int i = 0; i < Point::Dim; ++i) {
      p(i) = dist(e2);
    }
  }

  return random;
}

//! \brief Generates \p n random points uniformly distributed over a box of size
//! \p size.
template <typename Point>
inline std::vector<Point> GenerateRandomN(
    int n, typename Point::ScalarType size) {
  return GenerateRandomN<Point>(n, typename Point::ScalarType(0.0), size);
}
