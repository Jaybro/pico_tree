#pragma once

#include <cassert>
#include <opencv2/core.hpp>

#include "map_traits.hpp"

//! \file opencv_traits.hpp
//! \brief Contains traits that provide OpenCV support for PicoTree.
//! \details The following is supported:
//! * cv::Vec_<> as a point type.
//! * cv::Mat as a space type.

namespace pico_tree {

//! \brief point_traits provides an interface for cv::Point_<>.
//! \details point_traits<cv::Point_<Scalar_>> violates the strict aliasing rule
//! by interpreting a struct of scalars as an array of scalars and using this
//! specialization is therefore UB. Note that this specialization will work in
//! practice but you have been warned. Don't use it to avoid UB.
template <typename Scalar_>
struct point_traits<cv::Point_<Scalar_>> {
  static_assert(sizeof(cv::Point_<Scalar_>) == sizeof(Scalar_[2]), "");

  //! \brief Supported point type.
  using point_type = cv::Point_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = 2;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static scalar_type const* data(cv::Point_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point_.
  static constexpr size_type size(cv::Point_<Scalar_> const&) { return dim; }
};

//! \brief point_traits provides an interface for cv::Point3_<>.
//! \details point_traits<cv::Point3_<Scalar_>> violates the strict aliasing
//! rule by interpreting a struct of scalars as an array of scalars and using
//! this specialization is therefore UB. Note that this specialization will work
//! in practice but you have been warned. Don't use it to avoid UB.
template <typename Scalar_>
struct point_traits<cv::Point3_<Scalar_>> {
  static_assert(sizeof(cv::Point3_<Scalar_>) == sizeof(Scalar_[3]), "");

  //! \brief Supported point type.
  using point_type = cv::Point3_<Scalar_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = 3;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Point3_<Scalar_> const& point) {
    return &point.x;
  }

  //! \brief Returns the spatial dimension of a cv::Point3_.
  static constexpr size_type size(cv::Point3_<Scalar_> const&) { return dim; }
};

//! \brief point_traits provides an interface for cv::Vec<>.
template <typename Scalar_, int Dim_>
struct point_traits<cv::Vec<Scalar_, Dim_>> {
  //! \brief Supported point type.
  using point_type = cv::Vec<Scalar_, Dim_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = static_cast<size_type>(Dim_);

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static Scalar_ const* data(cv::Vec<Scalar_, Dim_> const& point) {
    return point.val;
  }

  //! \brief Returns the spatial dimension of a cv::Vec.
  static constexpr size_type size(cv::Vec<Scalar_, Dim_> const&) { return dim; }
};

//! \brief The opencv_mat_wrapper class provides compile time properties to a
//! cv::Mat.
template <typename Scalar_, size_t Dim_>
class opencv_mat_wrapper {
 public:
  using scalar_type = Scalar_;
  using size_type = size_t;
  static constexpr size_type dim = Dim_;
  using point_type = point_map<Scalar_ const, Dim_>;

  inline opencv_mat_wrapper(cv::Mat mat)
      : mat_(mat),
        size_(static_cast<size_type>(mat_.rows)),
        sdim_(mat_.step1()) {
    if constexpr (dim != dynamic_size) {
      assert(dim == sdim_);
    }
  }

  inline point_type operator[](int i) const {
    if constexpr (dim != dynamic_size) {
      return {data(i)};
    } else {
      return {data(i), sdim()};
    }
  }

  inline operator cv::Mat const&() const { return mat_; }

  inline operator cv::Mat&() { return mat_; }

  inline scalar_type const* data(int i) const {
    return mat_.template ptr<Scalar_>(i);
  }

  inline size_type size() const { return size_; }

  inline constexpr size_type sdim() const {
    if constexpr (dim != dynamic_size) {
      return dim;
    } else {
      return sdim_;
    }
  }

  inline cv::Mat const& mat() const { return mat_; }

  inline cv::Mat& mat() { return mat_; }

 private:
  cv::Mat mat_;
  size_type size_;
  size_type sdim_;
};

//! \brief Provides an interface for cv::Mat. Each row is considered a point.
//! \tparam Scalar_ Point coordinate type.
//! \tparam Dim_ The spatial dimension of each point. Set to
//! pico_tree::kDynamicSize when the dimension is only known at run-time.
template <typename Scalar_, size_t Dim_>
struct space_traits<opencv_mat_wrapper<Scalar_, Dim_>> {
  //! \brief The space_type of these traits.
  using space_type = opencv_mat_wrapper<Scalar_, Dim_>;
  //! \brief The point type used by space_type.
  using point_type = point_map<Scalar_ const, Dim_>;
  //! \brief The scalar type of point coordinates.
  using scalar_type = Scalar_;
  //! \brief The size and index type of point coordinates.
  using size_type = size_t;
  //! \brief Compile time spatial dimension.
  static constexpr size_type dim = Dim_;

  //! \brief Returns the point at \p idx from \p space.
  template <typename Index_>
  inline static point_type point_at(space_type const& space, Index_ idx) {
    return space[static_cast<int>(idx)];
  }

  //! \brief Returns number of points contained by \p space.
  inline static size_type size(space_type const& space) { return space.size(); }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  static constexpr size_type sdim(space_type const& space) {
    return space.sdim();
  }
};

}  // namespace pico_tree
