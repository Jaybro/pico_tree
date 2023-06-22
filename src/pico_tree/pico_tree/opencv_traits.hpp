#pragma once

#include <cassert>
#include <opencv2/core.hpp>

#include "map_traits.hpp"

//! \file opencv_traits.hpp
//! \brief Contains traits and classes that provide OpenCV support for PicoTree.
//! \details The following is supported:
//! * std::vector<cv::Point_<>> via pico_tree::StdTraits<>
//! * std::vector<cv::Point3_<>> via pico_tree::StdTraits<>
//! * std::vector<cv::Vec_<>> via pico_tree::StdTraits<>
//! * cv::Mat via pico_tree::CvTraits<>

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
  inline static ScalarType const* Coords(cv::Point_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point_.
  inline static SizeType constexpr Sdim(cv::Point_<Scalar_> const&) {
    return Dim;
  }
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
  inline static Scalar_ const* Coords(cv::Point3_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point3_.
  inline static SizeType constexpr Sdim(cv::Point3_<Scalar_> const&) {
    return Dim;
  }
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
  inline static Scalar_ const* Coords(cv::Vec<Scalar_, Dim_> const& point) {
    return point.val;
  }

  //! \brief Returns the spatial dimension of a cv::Vec.
  inline static SizeType constexpr Sdim(cv::Vec<Scalar_, Dim_> const&) {
    return Dim;
  }
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

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType Sdim(cv::Mat const& space) {
    if constexpr (Dim != kDynamicSize) {
      return Dim;
    } else {
      // TODO The use of step1() is actually quite expensive. Perhaps there is
      // an alternative.
      return space.step1();
    }
  }

  //! \brief Returns number of points contained by \p space.
  inline static SizeType Npts(cv::Mat const& space) {
    return static_cast<SizeType>(space.rows);
  }

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
};

}  // namespace pico_tree
