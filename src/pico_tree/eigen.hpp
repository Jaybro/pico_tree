#pragma once

namespace pico_tree {

namespace internal {

//! \private
template <typename Index, typename Matrix, bool RowMajor>
class EigenAdaptorBase;

//! \brief ColMajor EigenAdaptor.
template <typename Index, typename Matrix>
class EigenAdaptorBase<Index, Matrix, false> {
 public:
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Matrix::RowsAtCompileTime;
  //! \brief RowMajor flag that is true if the data is row-major.
  static constexpr bool RowMajor = false;

  //! \brief Constructs an EigenAdaptorBase from \p matrix. The matrix must be
  //! of dynamic size.
  //! \details To prevent an unwanted (deep) copy:
  //! \li Move the matrix inside the adaptor.
  //! \li Copy or move an Eigen::Map. Note that an Eigen::Map can be used as a
  //! proxy for a matrix.
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
template <typename Index, typename Matrix>
class EigenAdaptorBase<Index, Matrix, true> {
 public:
  //! \brief Spatial dimension. Eigen::Dynamic equals pico_tree::kDynamicDim.
  static constexpr int Dim = Matrix::ColsAtCompileTime;
  //! \brief RowMajor flag that is true if the data is row-major.
  static constexpr bool RowMajor = true;

  //! \brief Constructs an EigenAdaptorBase from \p matrix. The matrix must be
  //! of dynamic size.
  //! \details To prevent an unwanted (deep) copy:
  //! \li Move the matrix inside the adaptor.
  //! \li Copy or move an Eigen::Map. Note that an Eigen::Map can be used as a
  //! proxy for a matrix.
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
  // Fixed size Eigen members may need to be aligned in memory:
  // 1) As members are aligned with respect to the containing class the
  // EigenAdaptor would need to be aligned as well.
  // 2) Aligned members cannot be copied or moved (could be misaligned again
  // after a copy). This requires the interface to work quite differently for
  // fixed size matrices as compared to dynamic ones.
  // Combined with the idea that fixed size matrices are mostly useful when they
  // are small it is also not really worth it to support those through the
  // adaptor.
  static_assert(
      Matrix::RowsAtCompileTime == Eigen::Dynamic ||
          Matrix::ColsAtCompileTime == Eigen::Dynamic,
      "EIGEN_ADAPTOR_DOES_NOT_SUPPORT_FIXED_SIZE_MATRICES");

  //! \private
  using internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor>::
      EigenAdaptorBase;
  //! \private
  using internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor>::matrix_;

  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = typename Matrix::Scalar;

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
