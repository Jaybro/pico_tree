#pragma once

#include <cassert>
#include <opencv2/core.hpp>

#include "point_traits.hpp"

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

//! \brief A wrapper class for storing the row of a cv::Mat.
//! \details This wrapper is used to support compile and run time dimensions.
//! Storing a pointer and reference is a smaller footprint than the cv::Mat of a
//! row.
template <typename Scalar_, Size Dim_>
class CvMatRow {
 public:
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = Dim_;

  //! \brief Constructs a CvMatRow given the index \p idx of a row and a matrix
  //! \p space.
  inline CvMatRow(int idx, cv::Mat const& space)
      : coords_(space.ptr<Scalar_>(idx)), space_(space) {}

  //! \brief Returns a pointer to this point's coordinates.
  inline Scalar_ const* coords() const { return coords_; }

  //! \brief Returns the spatial dimension of this point.
  inline SizeType sdim() const {
    assert(Dim == kDynamicSize || Dim == space_.step1());
    // TODO This run time version is actually quite expensive when used. Perhaps
    // there is an alternative.
    return space_.step1();
  }

 private:
  Scalar_ const* coords_;
  cv::Mat const& space_;
};

//! \brief PointTraits provides an interface for CvMatRow<>.
template <typename Scalar_, Size Dim_>
struct PointTraits<CvMatRow<Scalar_, Dim_>> {
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static constexpr SizeType Dim = Dim_;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* Coords(CvMatRow<Scalar_, Dim_> const& point) {
    return point.coords();
  }

  //! \brief Returns the spatial dimension of \p point.
  inline static SizeType Sdim(CvMatRow<Scalar_, Dim_> const& point) {
    return point.sdim();
  }
};

//! \brief CvTraits provides an interface for cv::Mat. Each row is considered a
//! point.
//! \tparam Scalar_ Point coordinate type.
//! \tparam Dim_ The spatial dimension of each point. Set to
//! pico_tree::kDynamicSize when the dimension is only known at run-time.
template <typename Scalar_, int Dim_>
struct CvTraits {
  //! \brief The SpaceType of these traits.
  using SpaceType = cv::Mat;
  //! \brief The point type used by SpaceType.
  using PointType = CvMatRow<Scalar_, Dim_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief The index type of point coordinates.
  using IndexType = int;
  //! \brief Compile time spatial dimension.
  static constexpr SizeType Dim = Dim_;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType SpaceSdim(cv::Mat const& space) {
    assert(Dim == kDynamicSize || Dim == space.step1());
    // TODO This run time version is actually quite expensive when used. Perhaps
    // there is an alternative.
    return space.step1();
  }

  //! \brief Returns number of points contained by \p space.
  inline static IndexType SpaceNpts(cv::Mat const& space) { return space.rows; }

  //! \brief Returns the point at \p idx from \p space.
  inline static PointType PointAt(cv::Mat const& space, IndexType const idx) {
    return {idx, space};
  }

  //! \brief Returns the spatial dimension of \p point.
  //! \details Allowing the input type to be different from PointType gives us
  //! greater interfacing flexibility.
  template <typename OtherPoint>
  inline static SizeType PointSdim(OtherPoint const& point) {
    return PointTraits<OtherPoint>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  //! \details Allowing the input type to be different from PointType gives us
  //! greater interfacing flexibility.
  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return PointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pico_tree
