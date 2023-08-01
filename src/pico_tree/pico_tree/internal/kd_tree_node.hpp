#pragma once

namespace pico_tree::internal {

//!\brief Binary node base.
template <typename Derived>
struct KdTreeNodeBase {
  //! \brief Returns if the current node is a branch.
  inline bool IsBranch() const { return left != nullptr && right != nullptr; }
  //! \brief Returns if the current node is a leaf.
  inline bool IsLeaf() const { return left == nullptr && right == nullptr; }

  //! \brief Left child.
  Derived* left;
  //! \brief Right child.
  Derived* right;
};

//! \brief Tree leaf.
template <typename Index_>
struct KdTreeLeaf {
  //! \brief Begin of an index range.
  Index_ begin_idx;
  //! \brief End of an index range.
  Index_ end_idx;
};

//! \brief Tree branch.
template <typename Scalar_>
struct KdTreeBranchSplit {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
};

//! \brief Tree branch.
//! \details This branch version allows identifications (wrapping around) by
//! storing the boundaries of the box that corresponds to the current node. The
//! split value allows arbitrary splitting techniques.
template <typename Scalar_>
struct KdTreeBranchRange {
  //! \brief Split coordinate / index of the KdTree spatial dimension.
  int split_dim;
  //! \brief Minimum coordinate value of the left node box for split_dim.
  Scalar_ left_min;
  //! \brief Maximum coordinate value of the left node box for split_dim.
  Scalar_ left_max;
  //! \brief Minimum coordinate value of the right node box for split_dim.
  Scalar_ right_min;
  //! \brief Maximum coordinate value of the right node box for split_dim.
  Scalar_ right_max;
};

//! \brief NodeData is used to either store branch or leaf information. Which
//! union member is used can be tested with IsBranch() or IsLeaf().
template <typename Leaf, typename Branch>
union KdTreeNodeData {
  //! \brief Union branch data.
  Branch branch;
  //! \brief Union leaf data.
  Leaf leaf;
};

//! \brief KdTree node for a Euclidean space.
template <typename Index_, typename Scalar_>
struct KdTreeNodeEuclidean
    : public KdTreeNodeBase<KdTreeNodeEuclidean<Index_, Scalar_>> {
  using IndexType = Index_;
  using ScalarType = Scalar_;

  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index_>, KdTreeBranchSplit<Scalar_>> data;
};

//! \brief KdTree node for a topological space.
template <typename Index_, typename Scalar_>
struct KdTreeNodeTopological
    : public KdTreeNodeBase<KdTreeNodeTopological<Index_, Scalar_>> {
  using IndexType = Index_;
  using ScalarType = Scalar_;

  //! \brief Node data as a union of a leaf and branch.
  KdTreeNodeData<KdTreeLeaf<Index_>, KdTreeBranchRange<Scalar_>> data;
};

}  // namespace pico_tree::internal
