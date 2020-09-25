#pragma once

#include <Eigen/Dense>

namespace pico_tree {

namespace internal {

template <int Dims_>
struct EigenDimensions {
  inline static constexpr int Dims(int) { return Dims_; }
};

template <>
struct EigenDimensions<Eigen::Dynamic> {
  inline static int Dims(int dims) { return dims; }
};

template <typename Index, typename EigenMatrix, bool RowMajor>
class EigenAdaptorBase;

//! ColMajor EigenAdaptor.
template <typename Index_, typename EigenMatrix>
class EigenAdaptorBase<Index_, EigenMatrix, false> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dims = EigenMatrix::RowsAtCompileTime;
  static constexpr bool RowMajor = false;

  inline EigenAdaptorBase(EigenMatrix const& matrix) : matrix_(matrix) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, Index const dim) const {
    return matrix_(dim, idx);
  }

  //! Returns dimension \p dim of point \p point.
  template <typename Derived>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived> const& point, Index const dim) const {
    return point(dim);
  }

  //! Returns the amount of spatial dimensions of the points.
  inline Index num_dimensions() const {
    return EigenDimensions<Dims>::Dims(matrix_.rows());
  };

  //! Returns the number of points.
  inline Index num_points() const { return matrix_.cols(); };

 private:
  EigenMatrix matrix_;
};

//! RowMajor EigenAdaptor.
template <typename Index_, typename EigenMatrix>
class EigenAdaptorBase<Index_, EigenMatrix, true> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dims = EigenMatrix::ColsAtCompileTime;
  static constexpr bool RowMajor = true;

  inline EigenAdaptorBase(EigenMatrix const& matrix) : matrix_(matrix) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, Index const dim) const {
    return matrix_(idx, dim);
  }

  //! Returns dimension \p dim of point \p point.
  template <typename Derived>
  inline Scalar operator()(
      Eigen::MatrixBase<Derived> const& point, Index const dim) const {
    return point(dim);
  }

  //! Returns the amount of spatial dimensions of the points.
  inline Index num_dimensions() const {
    return EigenDimensions<Dims>::Dims(matrix_.cols());
  };

  //! Returns the number of points.
  inline Index num_points() const { return matrix_.rows(); };

 private:
  EigenMatrix matrix_;
};

}  // namespace internal

//! Adapts Eigen matrices so they can be used with any of the pico trees.
template <typename Index, typename EigenMatrix>
class EigenAdaptor
    : public internal::
          EigenAdaptorBase<Index, EigenMatrix, EigenMatrix::IsRowMajor> {
 public:
  inline EigenAdaptor(EigenMatrix const& matrix)
      : internal::EigenAdaptorBase<Index, EigenMatrix, EigenMatrix::IsRowMajor>(
            matrix) {}
};

}  // namespace pico_tree