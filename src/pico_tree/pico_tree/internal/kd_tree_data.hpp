#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/stream.hpp"

namespace pico_tree::internal {

//! \brief The data structure that represents a KdTree.
template <typename Node_, Size Dim_>
class KdTreeData {
 public:
  using IndexType = typename Node_::IndexType;
  using ScalarType = typename Node_::ScalarType;
  static Size constexpr Dim = Dim_;
  using BoxType = internal::Box<ScalarType, Dim>;
  using NodeType = Node_;
  using NodeAllocatorType = ChunkAllocator<NodeType, 256>;

  static KdTreeData Load(internal::Stream& stream) {
    typename BoxType::SizeType sdim;
    stream.Read(sdim);

    KdTreeData kd_tree_data{{}, BoxType(sdim), NodeAllocatorType(), nullptr};
    kd_tree_data.Read(stream);

    return kd_tree_data;
  }

  static void Save(KdTreeData const& data, internal::Stream& stream) {
    // Write sdim.
    stream.Write(data.root_box.size());
    data.Write(stream);
  }

  //! \brief Sorted indices that refer to points inside points_.
  std::vector<IndexType> indices;
  //! \brief Bounding box of the root node.
  BoxType root_box;
  //! \brief Memory allocator for tree nodes.
  NodeAllocatorType allocator;
  //! \brief Root of the KdTree.
  NodeType* root_node;

 private:
  //! \brief Recursively reads the Node and its descendants.
  inline NodeType* ReadNode(internal::Stream& stream) {
    NodeType* node = allocator.Allocate();
    bool is_leaf;
    stream.Read(is_leaf);

    if (is_leaf) {
      stream.Read(node->data.leaf);
      node->left = nullptr;
      node->right = nullptr;
    } else {
      stream.Read(node->data.branch);
      node->left = ReadNode(stream);
      node->right = ReadNode(stream);
    }

    return node;
  }

  //! \brief Recursively writes the Node and its descendants.
  inline void WriteNode(
      NodeType const* const node, internal::Stream& stream) const {
    if (node->IsLeaf()) {
      stream.Write(true);
      stream.Write(node->data.leaf);
    } else {
      stream.Write(false);
      stream.Write(node->data.branch);
      WriteNode(node->left, stream);
      WriteNode(node->right, stream);
    }
  }

  inline void Read(internal::Stream& stream) {
    stream.Read(indices);
    // The root box gets the correct size from the KdTree constructor.
    stream.Read(root_box.size(), root_box.min());
    stream.Read(root_box.size(), root_box.max());
    root_node = ReadNode(stream);
  }

  inline void Write(internal::Stream& stream) const {
    stream.Write(indices);
    stream.Write(root_box.min(), root_box.size());
    stream.Write(root_box.max(), root_box.size());
    WriteNode(root_node, stream);
  }
};

}  // namespace pico_tree::internal
