#pragma once

#include <Eigen/Core>

#include "core.hpp"

namespace pico_tree {

namespace internal {

template <typename Derived>
struct is_matrix_base
    : public std::is_base_of<
          Eigen::MatrixBase<typename std::decay<Derived>::type>,
          typename std::decay<Derived>::type> {};

template <typename Derived, bool IsRowMajor>
struct EigenVectorDim;

template <typename Derived>
struct EigenVectorDim<Derived, false> {
  static_assert(
      is_matrix_base<Derived>::value, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  static_assert(
      Derived::ColsAtCompileTime == 1, "DERIVED_TYPE_IS_NOT_A_VECTOR");
  static int constexpr Dim = Derived::RowsAtCompileTime;
};

template <typename Derived>
struct EigenVectorDim<Derived, true> {
  static_assert(
      is_matrix_base<Derived>::value, "DERIVED_TYPE_IS_NOT_AN_EIGEN_MATRIX");
  static_assert(
      Derived::RowsAtCompileTime == 1, "DERIVED_TYPE_IS_NOT_A_VECTOR");
  static int constexpr Dim = Derived::ColsAtCompileTime;
};

template <typename Derived>
struct EigenPointTraits : public EigenVectorDim<Derived, Derived::IsRowMajor> {
  using ScalarType = typename std::decay<typename Derived::Scalar>::type;

  inline static typename Derived::Scalar const* Coords(
      Eigen::MatrixBase<Derived> const& matrix) {
    return matrix.derived().data();
  }

  inline static int Sdim(Eigen::MatrixBase<Derived> const& matrix) {
    return matrix.size();
  }
};

template <typename Derived>
struct EigenPointBase {
  //! \brief Scalar type.
  using ScalarType = typename std::decay<typename Derived::Scalar>::type;
  //! \brief Index type.
  using IndexType = int;

  template <typename OtherDerived>
  inline static int PointSdim(Eigen::MatrixBase<OtherDerived> const& point) {
    static_assert(
        std::is_same<
            typename std::decay<typename Derived::Scalar>::type,
            typename std::decay<typename OtherDerived::Scalar>::type>::value,
        "INCOMPATIBLE_SCALAR_TYPES");
    return EigenPointTraits<OtherDerived>::Sdim(point);
  }

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
template <typename Derived, bool RowMajor>
struct EigenTraitsImpl;

//! \brief Space and Point traits for ColMajor Eigen types.
template <typename Derived>
struct EigenTraitsImpl<Derived, false> : public EigenPointBase<Derived> {
  using typename EigenPointBase<Derived>::IndexType;
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Derived::RowsAtCompileTime;
  using PointType = Eigen::Block<Derived const, Dim, 1, true>;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return matrix.rows();
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
template <typename Derived>
struct EigenTraitsImpl<Derived, true> : public EigenPointBase<Derived> {
  using typename EigenPointBase<Derived>::IndexType;
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Derived::ColsAtCompileTime;
  using PointType = Eigen::Block<Derived const, 1, Dim, true>;

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline static int SpaceSdim(Eigen::MatrixBase<Derived> const& matrix) {
    return matrix.cols();
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
template <typename Derived>
struct EigenTraitsBase : public EigenTraitsImpl<Derived, Derived::IsRowMajor> {
  using SpaceType = Derived;
};

}  // namespace internal

template <typename Derived>
struct EigenTraits;

template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_>
struct EigenTraits<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
    : internal::EigenTraitsBase<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> {
};

template <
    typename Scalar_,
    int Rows_,
    int Cols_,
    int Options_,
    int MaxRows_,
    int MaxCols_,
    int MapOptions_,
    typename StrideType_>
struct EigenTraits<Eigen::Map<
    Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
    MapOptions_,
    StrideType_>>
    : internal::EigenTraitsBase<Eigen::Map<
          Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>,
          MapOptions_,
          StrideType_>> {};

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
  //! \brief Creates an EigenL1.
  inline explicit EigenL1(int const) {}

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
  //! \brief Creates an L2Squared given a spatial dimension.
  inline explicit EigenL2(int const) {}

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
  //! \brief Creates an EigenL2Squared.
  inline explicit EigenL2Squared(int const) {}

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
