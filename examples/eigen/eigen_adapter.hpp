#pragma once

namespace internal {

template <int Dims_>
struct EigenDimensions {
  static constexpr int Dims(int) { return Dims_; }
};

template <>
struct EigenDimensions<Eigen::Dynamic> {
  static int Dims(int dims) { return dims; }
};

template <typename EigenMatrix, typename Index, bool RowMajor>
class EigenAdapterBase;

//! ColMajor EigenAdapter.
template <typename EigenMatrix, typename Index_>
class EigenAdapterBase<EigenMatrix, Index_, false> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dims = EigenMatrix::RowsAtCompileTime;
  static constexpr bool RowMajor = false;

  EigenAdapterBase(EigenMatrix const& matrix) : matrix_(matrix) {}

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

//! RowMajor EigenAdapter.
template <typename EigenMatrix, typename Index_>
class EigenAdapterBase<EigenMatrix, Index_, true> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dims = EigenMatrix::ColsAtCompileTime;
  static constexpr bool RowMajor = true;

  EigenAdapterBase(EigenMatrix const& matrix) : matrix_(matrix) {}

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

template <typename EigenMatrix, typename Index>
class EigenAdapter
    : public internal::
          EigenAdapterBase<EigenMatrix, Index, EigenMatrix::IsRowMajor> {
 public:
  EigenAdapter(EigenMatrix const& matrix)
      : internal::EigenAdapterBase<EigenMatrix, Index, EigenMatrix::IsRowMajor>(
            matrix) {}
};
