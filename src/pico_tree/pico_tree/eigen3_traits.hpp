#pragma once

#include <Eigen/Core>

#include "core.hpp"
#include "point_traits.hpp"
#include "space_traits.hpp"

//! \file eigen3_traits.hpp
//! \brief Provides an interface for spaces and points when working with types
//! from Eigen3.
//! \details It supports SpaceTraits<> for dynamic matrices and maps of dynamic
//! matrices, but not for fixed size matrices or maps of those. Fixed size
//! matrices are mostly useful when they are small. See section "Fixed vs.
//! Dynamic size" of the following link:
//! * https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
//!
//! PointTraits<> are supported for any type of matrix or matrix map.

namespace pico_tree {

namespace internal {

//! \brief A trait that determines if Derived inherits from Eigen::MatrixBase<>.
template <typename Derived>
struct is_matrix_base : public std::is_base_of<
                            Eigen::MatrixBase<std::remove_cv_t<Derived>>,
                            std::remove_cv_t<Derived>> {};

template <typename T>
inline constexpr bool is_matrix_base_v = is_matrix_base<T>::value;

template <typename Derived>
constexpr Eigen::Index EigenVectorDim() {
  static_assert(
      (!Derived::IsRowMajor && Derived::ColsAtCompileTime == 1) ||
          (Derived::IsRowMajor && Derived::RowsAtCompileTime == 1),
      "DERIVED_TYPE_IS_NOT_A_VECTOR");
  return Derived::IsRowMajor ? Derived::ColsAtCompileTime
                             : Derived::RowsAtCompileTime;
}

constexpr Size EigenDimToPicoDim(Eigen::Index dim) {
  return dim == Eigen::Dynamic ? kDynamicSize : static_cast<Size>(dim);
}

//! \brief EigenPointTraits provides an interface for the different point types
//! that can be used with EigenTraits.
//! \details Unlike the specialization of PointTraits for Eigen types, the
//! internal implementation supports matrix expressions.
template <typename Derived>
struct EigenPointTraits {
  static_assert(
      is_matrix_base_v<Derived>, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  //! \brief Supported point type.
  using PointType = Derived;
  //! \brief The scalar type of point coordinates.
  using ScalarType = std::remove_cv_t<typename Derived::Scalar>;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(EigenVectorDim<Derived>());

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static ScalarType const* data(Derived const& point) {
    return point.derived().data();
  }

  //! \brief Returns the spatial dimension of \p point.
  inline static SizeType size(Derived const& point) {
    return static_cast<SizeType>(point.size());
  }
};

template <typename Derived>
struct EigenTraitsBase {
  static_assert(
      Derived::RowsAtCompileTime == Eigen::Dynamic ||
          Derived::ColsAtCompileTime == Eigen::Dynamic,
      "FIXED_SIZE_MATRICES_ARE_NOT_SUPPORTED");

  //! \brief The SpaceType of these traits.
  using SpaceType = Derived;
};

//! \brief Space and Point traits for Eigen types.
template <typename Derived, bool RowMajor = Derived::IsRowMajor>
struct EigenTraitsImpl;

//! \brief Space and Point traits for ColMajor Eigen types.
template <typename Derived>
struct EigenTraitsImpl<Derived, false> : public EigenTraitsBase<Derived> {
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(Derived::RowsAtCompileTime);
  //! \brief The point type used by Derived.
  using PointType =
      Eigen::Block<Derived const, Derived::RowsAtCompileTime, 1, true>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = std::remove_cv_t<typename Derived::Scalar>;

  //! \brief Returns the point at index \p idx.
  template <typename Index_>
  inline static PointType PointAt(Derived const& matrix, Index_ idx) {
    return matrix.col(static_cast<Eigen::Index>(idx));
  }

  //! \brief Returns the number of points.
  inline static SizeType size(Derived const& matrix) {
    return static_cast<SizeType>(matrix.cols());
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static SizeType sdim(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<SizeType>(matrix.rows());
  }
};

//! \brief Space and Point traits for RowMajor Eigen types.
template <typename Derived>
struct EigenTraitsImpl<Derived, true> : public EigenTraitsBase<Derived> {
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(Derived::ColsAtCompileTime);
  //! \brief The point type used by Derived.
  using PointType =
      Eigen::Block<Derived const, 1, Derived::ColsAtCompileTime, true>;
  //! \brief The scalar type of point coordinates.
  using ScalarType = std::remove_cv_t<typename Derived::Scalar>;

  //! \brief Returns the point at index \p idx.
  template <typename Index_>
  inline static PointType PointAt(Derived const& matrix, Index_ idx) {
    return matrix.row(static_cast<Eigen::Index>(idx));
  }

  //! \brief Returns the number of points.
  inline static SizeType size(Derived const& matrix) {
    return static_cast<SizeType>(matrix.rows());
  }

  //! \brief Returns the number of coordinates or spatial dimension of each
  //! point.
  inline static SizeType sdim(Derived const& matrix) {
    return static_cast<SizeType>(matrix.cols());
  }
};

}  // namespace internal

//! \brief EigenTraits provides an interface for Eigen::Matrix<>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct SpaceTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : public internal::EigenTraitsImpl<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

//! \brief EigenTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct SpaceTraits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : public internal::EigenTraitsImpl<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

//! \brief PointTraits provides an interface for Eigen::Matrix<>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct PointTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : public internal::EigenPointTraits<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

//! \brief PointTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct PointTraits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : public internal::EigenPointTraits<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

//! \brief PointTraits provides an interface for Eigen::Block<>.
template <typename XprType_, int BlockRows_, int BlockCols_, bool InnerPanel_>
struct PointTraits<Eigen::Block<XprType_, BlockRows_, BlockCols_, InnerPanel_>>
    : public internal::EigenPointTraits<
          Eigen::Block<XprType_, BlockRows_, BlockCols_, InnerPanel_>> {};

}  // namespace pico_tree
