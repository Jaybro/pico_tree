#pragma once

#include "cover_tree_node.hpp"
#include "static_buffer.hpp"

namespace pico_tree::internal {

//! \brief The data structure that represents a cover_tree.
template <typename Index_, typename Scalar_>
class cover_tree_data {
 public:
  using index_type = Index_;
  using scalar_type = Scalar_;
  using node_type = cover_tree_node<Index_, Scalar_>;
  using node_allocator_type = internal::static_buffer<node_type>;

  //! \brief Memory allocator for tree nodes.
  node_allocator_type allocator;
  //! \brief Root of the cover_tree.
  node_type* root_node;
};

}  // namespace pico_tree::internal
