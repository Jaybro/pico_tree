#pragma once

#include <cassert>
#include <opencv2/core.hpp>

#include "map_traits.hpp"

//! \file opencv_traits.hpp
//! \brief Contains traits that provide OpenCV support for PicoTree.
//! \details The following is supported:
//! * std::vector<cv::Point_<>>
//! * std::vector<cv::Point3_<>>
//! * std::vector<cv::Vec_<>>
//! * cv::Mat

namespace pico_tree {

//! \brief PointTraits provides an interface for cv::Point_<>.
template <typename Scalar_>
struct PointTraits<cv::Point_<Scalar_>> {
  static_assert(sizeof(cv::Point_<Scalar_>) == (sizeof(Scalar_) * 2), "");

  //! \brief Supported point type.
  using PointType = cv::Point_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = 2;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static ScalarType const* data(cv::Point_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point_.
  static constexpr SizeType size(cv::Point_<Scalar_> const&) { return Dim; }
};

//! \brief PointTraits provides an interface for cv::Point3_<>.
template <typename Scalar_>
struct PointTraits<cv::Point3_<Scalar_>> {
  static_assert(sizeof(cv::Point3_<Scalar_>) == (sizeof(Scalar_) * 3), "");

  //! \brief Supported point type.
  using PointType = cv::Point3_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = 3;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Point3_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point3_.
  static constexpr SizeType size(cv::Point3_<Scalar_> const&) { return Dim; }
};

//! \brief PointTraits provides an interface for cv::Vec<>.
template <typename Scalar_, int Dim_>
struct PointTraits<cv::Vec<Scalar_, Dim_>> {
  //! \brief Supported point type.
  using PointType = cv::Vec<Scalar_, Dim_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = static_cast<SizeType>(Dim_);

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Vec<Scalar_, Dim_> const& point) {
    return point.val;
  }

  //! \brief Returns the spatial dimension of a cv::Vec.
  static constexpr SizeType size(cv::Vec<Scalar_, Dim_> const&) { return Dim; }
};

// TODO Support cv::Mat_ (instead?).
template <typename Scalar_, Size Dim_>
struct MatWrapper {
  inline MatWrapper(cv::Mat mat) : mat(mat) {}

  inline operator cv::Mat const&() const { return mat; }

  inline operator cv::Mat&() { return mat; }

  cv::Mat mat;
};

//! \brief Provides an interface for cv::Mat. Each row is considered a point.
//! \tparam Scalar_ Point coordinate type.
//! \tparam Dim_ The spatial dimension of each point. Set to
//! pico_tree::kDynamicSize when the dimension is only known at run-time.
template <typename Scalar_, Size Dim_>
struct SpaceTraits<MatWrapper<Scalar_, Dim_>> {
  //! \brief The SpaceType of these traits.
  using SpaceType = MatWrapper<Scalar_, Dim_>;
  //! \brief The point type used by SpaceType.
  using PointType = PointMap<Scalar_ const, Dim_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static constexpr SizeType Dim = Dim_;

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static PointType PointAt(cv::Mat const& space, Index_ idx) {
    if constexpr (Dim != kDynamicSize) {
      return {space.ptr<Scalar_>(static_cast<int>(idx))};
    } else {
      // TODO The use of step1() is actually quite expensive. Perhaps there is
      // an alternative.
      return {space.ptr<Scalar_>(static_cast<int>(idx)), space.step1()};
    }
  }

  //! \brief Returns number of points contained by \p space.
  inline static SizeType size(cv::Mat const& space) {
    return static_cast<SizeType>(space.rows);
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  static constexpr SizeType sdim(cv::Mat const& space) {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      // TODO The use of step1() is actually quite expensive. Perhaps there is
      // an alternative.
      return space.step1();
    }
  }
};

}  // namespace pico_tree
