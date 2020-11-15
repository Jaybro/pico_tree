#pragma once

namespace pico_tree {

namespace internal {

//! \private
template <typename Index, typename Matrix, bool RowMajor>
class EigenAdaptorBase;

//! \brief ColMajor EigenAdaptor.
template <typename Index_, typename Matrix>
class EigenAdaptorBase<Index_, Matrix, false> {
 public:
  //! \brief Index type.
  using Index = Index_;
  //! \brief Scalar type.
  using Scalar = typename Matrix::Scalar;
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Matrix::RowsAtCompileTime;
  //! \brief RowMajor flag that is true if the data is row-major.
  static constexpr bool RowMajor = false;

  //! \brief Default constructor. May be deleted if the Matrix doesn't have a
  //! default constructor.
  inline EigenAdaptorBase() = default;

  //! \brief Constructs an EigenAdaptorBase from \p matrix.
  //! \details To prevent an unwanted (deep) copy:
  //! \li Move the matrix if it has dynamic size.
  //! \li Copy or move an Eigen::Map.
  //! \li Use the default constructor to initialize a fixed size matrix.
  inline EigenAdaptorBase(Matrix matrix) : matrix_(std::move(matrix)) {}

  //! \brief Returns the point at index \p idx.
  inline Eigen::Block<Matrix const, Dim, 1, !RowMajor> const operator()(
      Index const idx) const {
    return matrix_.col(idx);
  }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return matrix_.rows(); }

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(matrix_.cols()); }

 protected:
  //! \private
  Matrix matrix_;
};

//! \brief RowMajor EigenAdaptor.
template <typename Index_, typename Matrix>
class EigenAdaptorBase<Index_, Matrix, true> {
 public:
  //! \brief Index type.
  using Index = Index_;
  //! \brief Scalar type.
  using Scalar = typename Matrix::Scalar;
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Matrix::ColsAtCompileTime;
  //! \brief RowMajor flag that is true if the data is row-major.
  static constexpr bool RowMajor = true;

  //! \brief Default constructor. May be deleted if the Matrix doesn't have a
  //! default constructor.
  inline EigenAdaptorBase() = default;

  //! \brief Constructs an EigenAdaptorBase from \p matrix.
  //! \details To prevent an unwanted (deep) copy:
  //! \li Move the matrix if it has dynamic size.
  //! \li Copy or move an Eigen::Map.
  //! \li Use the default constructor to initialize a fixed size matrix.
  inline EigenAdaptorBase(Matrix matrix) : matrix_(std::move(matrix)) {}

  //! \brief Returns the point at index \p idx.
  inline Eigen::Block<Matrix const, 1, Dim, RowMajor> const operator()(
      Index const idx) const {
    return matrix_.row(idx);
  }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return matrix_.cols(); }

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(matrix_.rows()); }

 protected:
  //! \private
  Matrix matrix_;
};

}  // namespace internal

//! Adapts Eigen matrices so they can be used with any of the pico trees.
template <typename Index, typename Matrix>
class EigenAdaptor
    : public internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor> {
 public:
  //! \private
  using Base = internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor>;
  using Base::EigenAdaptorBase;
  using Base::matrix_;

  //! \brief Returns a reference to the Eigen matrix.
  inline Matrix& matrix() { return matrix_; }

  //! \brief Returns a const reference to the Eigen matrix.
  inline Matrix const& matrix() const { return matrix_; }
};

//! \brief L1 metric using the L1 norm for measuring distances between points.
template <typename Scalar>
class EigenMetricL1 {
 public:
  //! \brief Creates an EigenMetricL1.
  inline EigenMetricL1(int const) {}

  //! \brief Calculates the distance between points \p p0 and \p p1.
  template <typename Derived0, typename Derived1>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived0> const& p0,
      Eigen::MatrixBase<Derived1> const& p1) const {
    static_assert(std::is_same<typename Derived0::Scalar, Scalar>::value);
    static_assert(std::is_same<typename Derived1::Scalar, Scalar>::value);
    return (p0 - p1).cwiseAbs().sum();
  }

  //! \brief Calculates the difference between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }
};

//! \brief The L2 metric measures distances between points using the squared L2
//! norm.
template <typename Scalar>
class EigenMetricL2 {
 public:
  //! \brief Creates an EigenMetricL2.
  inline EigenMetricL2(int const) {}

  //! \brief Calculates the distance between points \p p0 and \p p1.
  template <typename Derived0, typename Derived1>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived0> const& p0,
      Eigen::MatrixBase<Derived1> const& p1) const {
    static_assert(std::is_same<typename Derived0::Scalar, Scalar>::value);
    static_assert(std::is_same<typename Derived1::Scalar, Scalar>::value);
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
