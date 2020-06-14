#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "core.hpp"

namespace nano_tree {

template <typename Index, typename Scalar, int Dims, typename Points>
class KdTree {
 private:
  //! Node represents a tree node that is meant to be inherited.
  struct Node {
    union Data {
      struct Branch {
        Scalar split;
        Index dim;
      };

      struct Leaf {
        Index index;
      };

      Branch branch;
      Leaf leaf;
    };

    inline bool IsBranch() const { return left != nullptr && right != nullptr; }
    inline bool IsLeaf() const { return left == nullptr && right == nullptr; }

    Data data;
    Node* left;
    Node* right;
  };

 public:
  KdTree(Points const& points)
      : points_{points},
        dimensions_{points_.num_dimensions()},
        nodes_{internal::MaxNodesFromPoints(points_.num_points())},
        root_{MakeTree()} {
    assert(points_.num_points() > 0);
  }

 private:
  inline Node* MakeTree() {
    std::vector<Index> indices(points_.num_points());
    std::iota(indices.begin(), indices.end(), 0);
    return SplitIndices(0, points_.num_points(), 0, &indices);
  }

  inline Node* SplitIndices(
      Index const offset,
      Index const size,
      Index const dim,
      std::vector<Index>* p_indices) {
    std::vector<Index>& indices = *p_indices;
    Node* node = nodes_.MakeItem();
    //
    if (size == 1) {
      node->data.leaf.index = indices[offset];
      node->left = nullptr;
      node->right = nullptr;
    } else {
      Points const& points = points_;
      Index const left_size = size / 2;
      Index const right_size = size - left_size;
      Index const split = offset + left_size;
      std::nth_element(
          indices.begin() + offset,
          indices.begin() + split,
          indices.begin() + offset + size,
          [&points, dim](Index const a, Index const b) -> bool {
            return points(a, dim) < points(b, dim);
          });

      Index const next_dim = ((dim + 1) < dimensions_) ? (dim + 1) : 0;
      node->data.branch.split = points(indices[split], dim);
      node->data.branch.dim = dim;
      node->left = SplitIndices(offset, left_size, next_dim, p_indices);
      node->right = SplitIndices(split, right_size, next_dim, p_indices);
    }

    return node;
  }

  Points const& points_;
  Index const dimensions_;
  internal::ItemBuffer<Node> nodes_;
  Node* root_;
};

}  // namespace nano_tree