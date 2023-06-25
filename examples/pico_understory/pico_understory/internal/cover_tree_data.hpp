#pragma once

#include "cover_tree_node.hpp"
#include "static_buffer.hpp"

namespace pico_tree::internal {

//! \brief The data structure that represents a CoverTree.
template <typename Index_, typename Scalar_>
class CoverTreeData {
 public:
  using IndexType = Index_;
  using ScalarType = Scalar_;
  using NodeType = CoverTreeNode<Index_, Scalar_>;
  using NodeAllocatorType = internal::StaticBuffer<NodeType>;

  //! \brief Memory allocator for tree nodes.
  NodeAllocatorType allocator;
  //! \brief Root of the CoverTree.
  NodeType* root_node;
};

}  // namespace pico_tree::internal
