#pragma once

#include <vector>

namespace pico_tree::internal {

template <typename Index_, typename Scalar_>
struct CoverTreeNode {
  using IndexType = Index_;
  using ScalarType = Scalar_;

  inline bool IsBranch() const { return !children.empty(); }
  inline bool IsLeaf() const { return children.empty(); }

  // TODO Could be moved to the tree.
  ScalarType level;
  //! \brief Distance to the farthest child.
  ScalarType max_distance;
  IndexType index;
  std::vector<CoverTreeNode*> children;
};

}  // namespace pico_tree::internal
