#pragma once

#include <Eigen/Core>

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! \brief A trait that determines if Derived inherits from Eigen::MatrixBase<>.
template <typename Derived>
struct is_matrix_base
    : public std::is_base_of<
          Eigen::MatrixBase<typename std::decay<Derived>::type>,
          typename std::decay<Derived>::type> {};

//! \brief Dimension trait for Eigen vectors.
template <typename Derived, bool IsRowMajor>
struct EigenVectorDim;

//! \brief Provides compile time dimension for ColMajor Eigen types.
template <typename Derived>
struct EigenVectorDim<Derived, false> {
  static_assert(
      is_matrix_base<Derived>::value, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  static_assert(
      Derived::ColsAtCompileTime == 1, "DERIVED_TYPE_IS_NOT_A_VECTOR");
  //! \brief Compile time dimension of Derived.
  static int constexpr Dim = Derived::RowsAtCompileTime;
};

//! \brief Provides compile time dimension for RowMajor Eigen types.
template <typename Derived>
struct EigenVectorDim<Derived, true> {
  static_assert(
      is_matrix_base<Derived>::value, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  static_assert(
      Derived::RowsAtCompileTime == 1, "DERIVED_TYPE_IS_NOT_A_VECTOR");
  //! \brief Compile time dimension of Derived.
  static int constexpr Dim = Derived::ColsAtCompileTime;
};

//! \brief EigenPointTraits provides an interface for the different point types
//! that can be used with EigenTraits.
//! \details Unlike the specialization of StdPointTraits for Eigen types,
//! internal implementation supports matrix expressions.
template <typename Derived>
struct EigenPointTraits : public EigenVectorDim<Derived, Derived::IsRowMajor> {
  //! \brief The scalar type of point coordinates.
  using ScalarType = typename std::decay<typename Derived::Scalar>::type;

  //! \brief Returns a pointer to the coordinates of \p point.
  inline static typename Derived::Scalar const* Coords(
      Eigen::MatrixBase<Derived> const& point) {
    return point.derived().data();
  }

  //! \brief Returns the spatial dimension of \p point.
  inline static int Sdim(Eigen::MatrixBase<Derived> const& point) {
    return static_cast<int>(point.size());
  }
};

//! \brief Provides the Point interface for the different implementations of
//! EigenTraitsImpl.
template <typename Derived>
struct EigenPointBase {
  //! \brief The scalar type of point coordinates.
  using ScalarType = typename std::decay<typename Derived::Scalar>::type;

  //! \brief Returns the spatial dimension of \p point.
  template <typename OtherDerived>
  inline static int PointSdim(Eigen::MatrixBase<OtherDerived> const& point) {
    static_assert(
        std::is_same<
            typename std::decay<typename Derived::Scalar>::type,
            typename std::decay<typename OtherDerived::Scalar>::type>::value,
        "INCOMPATIBLE_SCALAR_TYPES");
    return EigenPointTraits<OtherDerived>::Sdim(point);
  }

  //! \brief Returns a pointer to the coordinates of \p point.
  template <typename OtherDerived>
  inline static ScalarType const* PointCoords(
      Eigen::MatrixBase<OtherDerived> const& point) {
    static_assert(
        std::is_same<
            typename std::decay<typename Derived::Scalar>::type,
            typename std::decay<typename OtherDerived::Scalar>::type>::value,
        "INCOMPATIBLE_SCALAR_TYPES");
    return EigenPointTraits<OtherDerived>::Coords(point);
  }
};

//! \brief Space and Point traits for Eigen types.
template <typename Derived, typename Index, bool RowMajor>
struct EigenTraitsImpl;

//! \brief Space and Point traits for ColMajor Eigen types.
template <typename Derived, typename Index>
struct EigenTraitsImpl<Derived, Index, false> : public EigenPointBase<Derived> {
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Derived::RowsAtCompileTime;
  //! \brief The point type used by Derived.
  using PointType = Eigen::Block<Derived const, Dim, 1, true>;
  //! \brief The index type of point coordinates.
  using IndexType = Index;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<int>(matrix.rows());
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
template <typename Derived, typename Index>
struct EigenTraitsImpl<Derived, Index, true> : public EigenPointBase<Derived> {
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Derived::ColsAtCompileTime;
  //! \brief The point type used by Derived.
  using PointType = Eigen::Block<Derived const, 1, Dim, true>;
  //! \brief The index type of point coordinates.
  using IndexType = Index;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return static_cast<int>(matrix.cols());
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

//! \private This struct simply reduces some of the template argument overhead.
template <typename Derived, typename Index>
struct EigenTraitsBase
    : public EigenTraitsImpl<Derived, Index, Derived::IsRowMajor> {
  using SpaceType = Derived;
};

}  // namespace internal

//! \brief EigenTraits provides an interface for spaces and points when working
//! with Eigen types.
//! \tparam Derived An Eigen::Matrix<> or Eigen::Map<Eigen::Matrix<>>.
//! \tparam Index Type used for indexing. Defaults to int.
template <typename Derived, typename Index = int>
struct EigenTraits;

//! \brief EigenTraits provides an interface for Eigen::Matrix<>.
//! \tparam Index Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    typename Index>
struct EigenTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    Index>
    : internal::EigenTraitsBase<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          Index> {};

//! \brief EigenTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
//! \tparam Index Type used for indexing. Defaults to int.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_,
    typename Index>
struct EigenTraits<
    Eigen::Map<
        Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
        MapOptions_,
        StrideType_>,
    Index>
    : internal::EigenTraitsBase<
          Eigen::Map<
              Eigen::
                  Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
              MapOptions_,
              StrideType_>,
          Index> {};

//! \brief StdPointTraits provides an interface for Eigen::Matrix<>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct StdPointTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : internal::EigenPointTraits<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

//! \brief StdPointTraits provides an interface for Eigen::Map<Eigen::Matrix<>>.
template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct StdPointTraits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : internal::EigenPointTraits<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

//! \brief EigenL1 metric for measuring the Taxicab or Manhattan distance
//! between points.
template <typename Scalar>
class EigenL1 {
 public:
  //! \brief Calculates the distance between points \p p0 and \p p1.
  template <typename Derived0, typename Derived1>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived0> const& p0,
      Eigen::MatrixBase<Derived1> const& p1) const {
    return (p0 - p1).cwiseAbs().sum();
  }

  //! \brief Calculates the difference between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }
};

//! \brief EigenL2 metric for measuring Euclidean distances between points.
template <typename Scalar>
class EigenL2 {
 public:
  //! \brief Calculates the distance between points \p p0 and \p p1.
  template <typename Derived0, typename Derived1>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived0> const& p0,
      Eigen::MatrixBase<Derived1> const& p1) const {
    return (p0 - p1).norm();
  }

  //! \brief Calculates the distance between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }
};

//! \brief The EigenL2Squared semimetric measures squared Euclidean distances
//! between points.
template <typename Scalar>
class EigenL2Squared {
 public:
  //! \brief Calculates the distance between points \p p0 and \p p1.
  template <typename Derived0, typename Derived1>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived0> const& p0,
      Eigen::MatrixBase<Derived1> const& p1) const {
    return (p0 - p1).squaredNorm();
  }

  //! \brief Calculates the difference between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared value of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }
};

}  // namespace pico_tree
