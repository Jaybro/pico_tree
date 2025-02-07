#pragma once

namespace pico_tree::internal {

//!\brief Binary node base.
template <typename Derived_>
struct kd_tree_node_base {
  //! \brief Returns if the current node is a branch.
  inline bool is_branch() const { return left != nullptr && right != nullptr; }
  //! \brief Returns if the current node is a leaf.
  inline bool is_leaf() const { return left == nullptr && right == nullptr; }

  //! \brief Left child.
  Derived_* left;
  //! \brief Right child.
  Derived_* right;
};

//! \brief Tree leaf.
template <typename Index_>
struct kd_tree_leaf {
  //! \brief Begin of an index range.
  Index_ begin_idx;
  //! \brief End of an index range.
  Index_ end_idx;
};

//! \brief Tree branch.
template <typename Scalar_>
struct kd_tree_branch_split {
  //! \brief Split coordinate / index of the kd_tree spatial dimension.
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
struct kd_tree_branch_range {
  //! \brief Split coordinate / index of the kd_tree spatial dimension.
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
//! union member is used can be tested with is_branch() or is_leaf().
template <typename Leaf_, typename Branch_>
union kd_tree_node_data {
  //! \brief Union branch data.
  Branch_ branch;
  //! \brief Union leaf data.
  Leaf_ leaf;
};

//! \brief kd_tree node for a Euclidean space.
template <typename Index_, typename Scalar_>
struct kd_tree_node_euclidean
    : public kd_tree_node_base<kd_tree_node_euclidean<Index_, Scalar_>> {
  using index_type = Index_;
  using scalar_type = Scalar_;

  template <typename Box_>
  inline void set_branch(
      Box_ const& left_box, Box_ const& right_box, size_t const split_dim) {
    data.branch.split_dim = static_cast<int>(split_dim);
    data.branch.left_max = left_box.max(split_dim);
    data.branch.right_min = right_box.min(split_dim);
  }

  //! \brief Node data as a union of a leaf and branch.
  kd_tree_node_data<kd_tree_leaf<Index_>, kd_tree_branch_split<Scalar_>> data;
};

//! \brief kd_tree node for a topological space.
template <typename Index_, typename Scalar_>
struct kd_tree_node_topological
    : public kd_tree_node_base<kd_tree_node_topological<Index_, Scalar_>> {
  using index_type = Index_;
  using scalar_type = Scalar_;

  template <typename Box_>
  inline void set_branch(
      Box_ const& left_box, Box_ const& right_box, size_t const split_dim) {
    data.branch.split_dim = static_cast<int>(split_dim);
    data.branch.left_min = left_box.min(split_dim);
    data.branch.left_max = left_box.max(split_dim);
    data.branch.right_min = right_box.min(split_dim);
    data.branch.right_max = right_box.max(split_dim);
  }

  //! \brief Node data as a union of a leaf and branch.
  kd_tree_node_data<kd_tree_leaf<Index_>, kd_tree_branch_range<Scalar_>> data;
};

}  // namespace pico_tree::internal
