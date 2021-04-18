#pragma once

#include <opencv2/core.hpp>

#include "std_traits.hpp"

//! \file opencv.hpp
//! \brief Contains traits and classes that provide OpenCV support for PicoTree.
//! \details The following is supported:
//! * std::vector<cv::Point_<>> via pico_tree::StdTraits<>
//! * std::vector<cv::Point3_<>> via pico_tree::StdTraits<>
//! * std::vector<cv::Vec_<>> via pico_tree::StdTraits<>
//! * cv::Mat via pico_tree::CvTraits<>

namespace pico_tree {

//! \brief StdPointTraits provides an interface for cv::Point_<>.
template <typename Scalar_>
struct StdPointTraits<cv::Point_<Scalar_>> {
  static_assert(sizeof(cv::Point_<Scalar_>) == (sizeof(Scalar_) * 2), "");

  using ScalarType = Scalar_;
  static constexpr int Dim = 2;

  inline static ScalarType const* Coords(cv::Point_<Scalar_> const& point) {
    return &point.x;
  }

  inline static int constexpr Sdim(cv::Point_<Scalar_> const&) { return Dim; }
};

//! \brief StdPointTraits provides an interface for cv::Point3_<>.
template <typename Scalar_>
struct StdPointTraits<cv::Point3_<Scalar_>> {
  static_assert(sizeof(cv::Point3_<Scalar_>) == (sizeof(Scalar_) * 3), "");

  using ScalarType = Scalar_;
  static constexpr int Dim = 3;

  inline static Scalar_ const* Coords(cv::Point3_<Scalar_> const& point) {
    return &point.x;
  }

  inline static int constexpr Sdim(cv::Point3_<Scalar_> const&) { return Dim; }
};

//! \brief StdPointTraits provides an interface for cv::Vec<>.
template <typename Scalar_, int Dim_>
struct StdPointTraits<cv::Vec<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline static Scalar_ const* Coords(cv::Vec<Scalar_, Dim_> const& point) {
    return point.val;
  }

  inline static int constexpr Sdim(cv::Vec<Scalar_, Dim_> const&) {
    return Dim;
  }
};

//! \brief A wrapper class for storing the row of a cv::Mat.
//! \details This wrapper is used to support compile and run time dimensions.
//! Storing a pointer and reference is a smaller footprint than the cv::Mat of a
//! row.
template <typename Scalar_, int Dim_>
class CvMatRow {
 public:
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief Compile time spatial dimension.
  static constexpr int Dim = Dim_;

  //! \brief Constructs a CvMatRow given the index \p idx of a row and a matrix
  //! \p space.
  inline CvMatRow(int idx, cv::Mat const& space)
      : coords_(space.ptr<Scalar_>(idx)), space_(space) {}

  //! \brief Returns a pointer to this point's coordinates.
  inline Scalar_ const* coords() const { return coords_; }

  //! \brief Returns the amount of spatial dimensions of this point.
  inline int sdim() const {
    // TODO This run time version is actually quite expensive when used. Perhaps
    // there is an alternative.
    return space_.step1();
  }

 private:
  Scalar_ const* coords_;
  cv::Mat const& space_;
};

//! \brief StdPointTraits provides an interface for CvMatRow<>.
template <typename Scalar_, int Dim_>
struct StdPointTraits<CvMatRow<Scalar_, Dim_>> {
  using ScalarType = Scalar_;
  static constexpr int Dim = Dim_;

  inline static Scalar_ const* Coords(CvMatRow<Scalar_, Dim_> const& point) {
    return point.coords();
  }

  inline static int Sdim(CvMatRow<Scalar_, Dim_> const& point) {
    return point.sdim();
  }
};

//! \brief CvTraits provides an interface for cv::Mat. Each row is considered a
//! point.
//! \tparam Scalar_ Point coordinate type.
//! \tparam Dim_ The spatial dimension of each point. Set to
//! pico_tree::kDynamicDim when the dimension is only known at run-time.
template <typename Scalar_, int Dim_>
struct CvTraits {
  //! \brief The SpaceType of these traits.
  using SpaceType = cv::Mat;
  //! \brief The point type used by SpaceType.
  using PointType = CvMatRow<Scalar_, Dim_>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = Scalar_;
  //! \brief The index type of point coordinates.
  using IndexType = int;
  //! \brief Compile time spatial dimension.
  static constexpr int Dim = Dim_;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int SpaceSdim(cv::Mat const& space) {
    return static_cast<int>(space.step1());
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
  inline static int PointSdim(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  //! \details Allowing the input type to be different from PointType gives us
  //! greater interfacing flexibility.
  template <typename OtherPoint>
  inline static ScalarType const* PointCoords(OtherPoint const& point) {
    return StdPointTraits<OtherPoint>::Coords(point);
  }
};

}  // namespace pico_tree
