#pragma once

namespace pico_tree {

namespace internal {

template <typename Index, typename EigenMatrix, bool RowMajor>
class EigenAdaptorBase;

//! ColMajor EigenAdaptor.
template <typename Index_, typename EigenMatrix>
class EigenAdaptorBase<Index_, EigenMatrix, false> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dim = EigenMatrix::RowsAtCompileTime;
  static constexpr bool RowMajor = false;

  inline EigenAdaptorBase(EigenMatrix const& matrix) : matrix_(matrix) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, Index const dim) const {
    return matrix_(dim, idx);
  }

  //! Returns dimension \p dim of point \p point.
  template <typename Point>
  inline Scalar operator()(Point const& point, Index const dim) const {
    return point(dim);
  }

  //! Returns the dimension of the space in which the points reside. I.e., the
  //! amount of coordinates each point has.
  inline int sdim() const { return matrix_.rows(); };

  //! Returns the number of points.
  inline Index npts() const { return matrix_.cols(); };

 private:
  EigenMatrix matrix_;
};

//! RowMajor EigenAdaptor.
template <typename Index_, typename EigenMatrix>
class EigenAdaptorBase<Index_, EigenMatrix, true> {
 public:
  using Index = Index_;
  using Scalar = typename EigenMatrix::Scalar;
  static constexpr int Dim = EigenMatrix::ColsAtCompileTime;
  static constexpr bool RowMajor = true;

  inline EigenAdaptorBase(EigenMatrix const& matrix) : matrix_(matrix) {}

  //! Returns dimension \p dim of point \p idx.
  inline Scalar operator()(Index const idx, Index const dim) const {
    return matrix_(idx, dim);
  }

  //! Returns dimension \p dim of point \p point.
  template <typename Point>
  inline Scalar operator()(Point const& point, Index const dim) const {
    return point(dim);
  }

  //! Returns the dimension of the space in which the points reside. I.e., the
  //! amount of coordinates each point has.
  inline int sdim() const { return matrix_.cols(); };

  //! Returns the number of points.
  inline Index npts() const { return matrix_.rows(); };

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
  using internal::EigenAdaptorBase<
      Index,
      EigenMatrix,
      EigenMatrix::IsRowMajor>::EigenAdaptorBase;
};

}  // namespace pico_tree
