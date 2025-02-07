#pragma once

#include <vector>

namespace pico_tree::internal {

template <typename Index_, typename Scalar_>
struct cover_tree_node {
  using index_type = Index_;
  using scalar_type = Scalar_;

  inline bool is_branch() const { return !children.empty(); }
  inline bool is_leaf() const { return children.empty(); }

  // TODO Could be moved to the tree.
  scalar_type level;
  //! \brief Distance to the farthest child.
  scalar_type max_distance;
  index_type index;
  std::vector<cover_tree_node*> children;
};

}  // namespace pico_tree::internal
