#pragma once

#include "pico_tree/core.hpp"
#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/stream_wrapper.hpp"

namespace pico_tree::internal {

//! \brief The data structure that represents a kd_tree.
template <typename Node_, size_t Dim_>
class kd_tree_data {
 public:
  using index_type = typename Node_::index_type;
  using scalar_type = typename Node_::scalar_type;
  static size_t constexpr dim = Dim_;
  using box_type = internal::box<scalar_type, dim>;
  using node_type = Node_;
  using node_allocator_type = chunk_allocator<node_type, 256>;

  static kd_tree_data load(stream_wrapper& stream) {
    typename box_type::size_type sdim;
    stream.read(sdim);

    kd_tree_data kd_tree_data{
        {}, box_type(sdim), node_allocator_type(), nullptr};
    kd_tree_data.read(stream);

    return kd_tree_data;
  }

  static void save(kd_tree_data const& data, stream_wrapper& stream) {
    // Write sdim.
    stream.write(data.root_box.size());
    data.write(stream);
  }

  //! \brief Sorted indices that refer to points inside points_.
  std::vector<index_type> indices;
  //! \brief Bounding box of the root node.
  box_type root_box;
  //! \brief Memory allocator for tree nodes.
  node_allocator_type allocator;
  //! \brief Root of the kd_tree.
  node_type* root_node;

 private:
  //! \brief Recursively reads the Node and its descendants.
  inline node_type* read_node(stream_wrapper& stream) {
    node_type* node = allocator.allocate();
    bool is_leaf;
    stream.read(is_leaf);

    if (is_leaf) {
      stream.read(node->data.leaf);
      node->left = nullptr;
      node->right = nullptr;
    } else {
      stream.read(node->data.branch);
      node->left = read_node(stream);
      node->right = read_node(stream);
    }

    return node;
  }

  //! \brief Recursively writes the Node and its descendants.
  inline void write_node(
      node_type const* const node, stream_wrapper& stream) const {
    if (node->is_leaf()) {
      stream.write(true);
      stream.write(node->data.leaf);
    } else {
      stream.write(false);
      stream.write(node->data.branch);
      write_node(node->left, stream);
      write_node(node->right, stream);
    }
  }

  inline void read(stream_wrapper& stream) {
    stream.read(indices);
    // The root box gets the correct size from the kd_tree constructor.
    stream.read(root_box.size(), root_box.min());
    stream.read(root_box.size(), root_box.max());
    root_node = read_node(stream);
  }

  inline void write(stream_wrapper& stream) const {
    stream.write(indices);
    stream.write(root_box.min(), root_box.size());
    stream.write(root_box.max(), root_box.size());
    write_node(root_node, stream);
  }
};

}  // namespace pico_tree::internal
