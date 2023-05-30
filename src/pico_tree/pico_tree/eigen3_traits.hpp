#pragma once

#include <Eigen/Core>

#include "metric.hpp"
#include "point_traits.hpp"

//! \file eigen3_traits.hpp
//! \brief Contains traits and classes that provide Eigen3 support for PicoTree.

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
  //! \brief The scalar type of point coordinates.
  using ScalarType = std::remove_cv_t<typename Derived::Scalar>;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Compile time spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(EigenVectorDim<Derived>());

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static ScalarType const* Coords(
      Eigen::MatrixBase<Derived> const& point) {
    return point.derived().data();
  }

  //! \brief Returns the spatial dimension of \p point.
  inline static SizeType Sdim(Eigen::MatrixBase<Derived> const& point) {
    return static_cast<SizeType>(point.size());
  }
};

//! \brief Space and Point traits for Eigen types.
template <typename Derived, typename Index_, bool RowMajor>
struct EigenTraitsImpl;

//! \brief Space and Point traits for ColMajor Eigen types.
template <typename Derived, typename Index_>
struct EigenTraitsImpl<Derived, Index_, false> {
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(Derived::RowsAtCompileTime);
  //! \brief The point type used by Derived.
  using PointType =
      Eigen::Block<Derived const, Derived::RowsAtCompileTime, 1, true>;
  //! \brief The index type of point coordinates.
  using IndexType = Index_;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<SizeType>(matrix.rows());
  }

  //! \brief Returns the number of points.
  inline static IndexType SpaceNpts(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<IndexType>(matrix.cols());
  }

  //! \brief Returns the point at index \p idx.
  inline static PointType PointAt(
      Eigen::MatrixBase<Derived> const& matrix, IndexType const idx) {
    return matrix.col(idx);
  }
};

//! \brief Space and Point traits for RowMajor Eigen types.
template <typename Derived, typename Index_>
struct EigenTraitsImpl<Derived, Index_, true> {
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;
  //! \brief Spatial dimension.
  static SizeType constexpr Dim = EigenDimToPicoDim(Derived::ColsAtCompileTime);
  //! \brief The point type used by Derived.
  using PointType =
      Eigen::Block<Derived const, 1, Derived::ColsAtCompileTime, true>;
  //! \brief The index type of point coordinates.
  using IndexType = Index_;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static SizeType SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<SizeType>(matrix.cols());
  }

  //! \brief Returns the number of points.
  inline static IndexType SpaceNpts(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<IndexType>(matrix.rows());
  }

  //! \brief Returns the point at index \p idx.
  inline static PointType PointAt(
      Eigen::MatrixBase<Derived> const& matrix, IndexType const idx) {
    return matrix.row(idx);
  }
};

//! \brief This struct simply reduces some of the template argument overhead.
template <typename Derived, typename Index_>
struct EigenTraitsBase
    : public EigenTraitsImpl<Derived, Index_, Derived::IsRowMajor> {
  static_assert(
      Derived::RowsAtCompileTime == Eigen::Dynamic ||
          Derived::ColsAtCompileTime == Eigen::Dynamic,
      "FIXED_SIZE_MATRICES_ARE_NOT_SUPPORTED");

  //! \brief The SpaceType of these traits.
  using SpaceType = Derived;
  //! \brief The scalar type of point coordinates.
  using ScalarType = std::remove_cv_t<typename Derived::Scalar>;
  //! \brief The size and index type of point coordinates.
  using SizeType = Size;

  //! \brief Returns the spatial dimension of \p point.
  template <typename OtherDerived>
  inline static SizeType PointSdim(
      Eigen::MatrixBase<OtherDerived> const& point) {
    static_assert(
        std::is_same_v<
            ScalarType,
            std::remove_cv_t<typename OtherDerived::Scalar>>,
        "INCOMPATIBLE_SCALAR_TYPES");
    return EigenPointTraits<OtherDerived>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  template <typename OtherDerived>
  inline static ScalarType const* PointCoords(
      Eigen::MatrixBase<OtherDerived> const& point) {
    static_assert(
        std::is_same_v<
            ScalarType,
            std::remove_cv_t<typename OtherDerived::Scalar>>,
        "INCOMPATIBLE_SCALAR_TYPES");
    return EigenPointTraits<OtherDerived>::Coords(point);
  }
};

}  // namespace internal

//! \brief EigenTraits provides an interface for spaces and points when working
//! with Eigen types.
//! \details It supports dynamic matrices and maps of dynamic matrices. Support
//! for fixed size matrices or maps of those is disabled. Fixed size matrices
//! are mostly useful when they are small. See section "Fixed vs. Dynamic size"
//! of the following link:
//! * https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html
//! <p/>
//! Special care needs to be taken to work with fixed size matrices as well,
//! adding to the complexity of this library with very little in return. This
//! results in the choice to not support them through EigenTraits.
//! <p/>
//! Special care:
//! * Aligned members of fixed size cannot be copied or moved (a move is a
//! copy). https://eigen.tuxfamily.org/dox/group__TopicPassingByValue.html
//! * They may need to be aligned in memory. As members are aligned with
//! respect to the containing class the EigenAdaptor would need to be aligned
//! as well.
//! https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
//! \tparam Derived An Eigen::Matrix<> or Eigen::Map<Eigen::Matrix<>>.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <typename Derived, typename Index_ = int>
struct EigenTraits;

//! \brief EigenTraits provides an interface for Eigen::Matrix<>.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    typename Index_>
struct EigenTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    Index_>
    : internal::EigenTraitsBase<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          Index_> {};

//! \brief EigenTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_,
    typename Index_>
struct EigenTraits<
    Eigen::Map<
        Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
        MapOptions_,
        StrideType_>,
    Index_>
    : internal::EigenTraitsBase<
          Eigen::Map<
              Eigen::
                  Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
              MapOptions_,
              StrideType_>,
          Index_> {};

//! \brief EigenTraits provides an interface for
//! std::reference_wrapper<Eigen::Matrix<>>.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    typename Index_>
struct EigenTraits<
    std::reference_wrapper<
        Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>,
    Index_>
    : public EigenTraits<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          Index_> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::reference_wrapper<
      Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>;
};

//! \brief EigenTraits provides an interface for
//! std::reference_wrapper<Eigen::Matrix<> const>.
//! \tparam Index_ Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    typename Index_>
struct EigenTraits<
    std::reference_wrapper<
        Eigen::
            Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const>,
    Index_>
    : public EigenTraits<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          Index_> {
  //! \brief The SpaceType of these traits.
  using SpaceType = std::reference_wrapper<
      Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_> const>;
};

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
    : internal::EigenPointTraits<
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
    : internal::EigenPointTraits<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

}  // namespace pico_tree
