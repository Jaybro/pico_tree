#pragma once

namespace pico_tree {

namespace internal {

template <typename Index, typename Matrix, bool RowMajor>
class EigenAdaptorBase;

//! \brief ColMajor EigenAdaptor.
template <typename Index_, typename Matrix>
class EigenAdaptorBase<Index_, Matrix, false> {
 public:
  using Index = Index_;
  using Scalar = typename Matrix::Scalar;
  static constexpr int Dim = Matrix::RowsAtCompileTime;
  static constexpr bool RowMajor = false;

  inline EigenAdaptorBase(Matrix const& matrix) : matrix_(matrix) {}

  //! \brief Returns the point at index \p idx.
  inline Eigen::Block<Matrix const, Dim, 1, !RowMajor> const operator()(
      Index const idx) const {
    return matrix_.col(idx);
  }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return matrix_.rows(); };

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(matrix_.cols()); };

 private:
  Matrix matrix_;
};

//! \brief RowMajor EigenAdaptor.
template <typename Index_, typename Matrix>
class EigenAdaptorBase<Index_, Matrix, true> {
 public:
  using Index = Index_;
  using Scalar = typename Matrix::Scalar;
  static constexpr int Dim = Matrix::ColsAtCompileTime;
  static constexpr bool RowMajor = true;

  inline EigenAdaptorBase(Matrix const& matrix) : matrix_(matrix) {}

  //! \brief Returns the point at index \p idx.
  inline Eigen::Block<Matrix const, 1, Dim, RowMajor> const operator()(
      Index const idx) const {
    return matrix_.row(idx);
  }

  //! \brief Returns the dimension of the space in which the points reside.
  //! I.e., the amount of coordinates each point has.
  inline int sdim() const { return matrix_.cols(); };

  //! \brief Returns the number of points.
  inline Index npts() const { return static_cast<Index>(matrix_.rows()); };

 private:
  Matrix matrix_;
};

}  // namespace internal

//! Adapts Eigen matrices so they can be used with any of the pico trees.
template <typename Index, typename Matrix>
class EigenAdaptor
    : public internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor> {
 public:
  using internal::EigenAdaptorBase<Index, Matrix, Matrix::IsRowMajor>::
      EigenAdaptorBase;
};

}  // namespace pico_tree
