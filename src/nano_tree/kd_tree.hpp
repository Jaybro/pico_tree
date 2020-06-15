#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "core.hpp"

namespace nano_tree {

//! L2 metric that calculates the squared distance between two points.
//! Can be replaced by a custom metric.
template <typename Index, typename Scalar, int Dims, typename Points>
class MetricL2 {
 public:
  MetricL2(Points const& points) : points_{points} {}

  //! Calculates the difference between two points given an query point and an
  //! index to a point.
  //! \tparam P Point type.
  //! \param p Point.
  //! \param idx Index.
  template <typename P>
  inline Scalar operator()(P const& p, Index idx) const {
    Scalar d{};

    for (Index i = 0;
         i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
         ++i) {
      d += std::pow(points_(p, i) - points_(idx, i), 2);
    }

    return d;
  }

 private:
  Points const& points_;
};

template <
    typename Index,
    typename Scalar,
    int Dims,
    typename Points,
    typename Metric = MetricL2<Index, Scalar, Dims, Points>>
class KdTree {
 private:
  struct Node {
    union Data {
      struct Branch {
        Scalar split;
        Index dim;
      };

      struct Leaf {
        Index idx;
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
        metric_{points_},
        dimensions_{points_.num_dimensions()},
        nodes_{internal::MaxNodesFromPoints(points_.num_points())},
        root_{MakeTree()} {
    assert(points_.num_points() > 0);
  }

  template <typename P>
  inline std::pair<Index, Scalar> SearchNn(P const& p) const {
    Index idx;
    Scalar min = std::numeric_limits<Scalar>::max();
    SearchNn(p, root_, &idx, &min);
    return {idx, min};
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
      node->data.leaf.idx = indices[offset];
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

  template <typename P>
  inline void SearchNn(
      P const& p, Node const* const node, Index* idx, Scalar* min) const {
    if (node->IsBranch()) {
      Scalar const v = points_(p, node->data.branch.dim);
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      if (v <= node->data.branch.split) {
        SearchNn(p, node->left, idx, min);
        if (*min > std::pow(v - node->data.branch.split, 2)) {
          SearchNn(p, node->right, idx, min);
        }
      } else {
        SearchNn(p, node->right, idx, min);
        if (*min > std::pow(node->data.branch.split - v, 2)) {
          SearchNn(p, node->left, idx, min);
        }
      }
    } else {
      Scalar d = metric_(p, node->data.leaf.idx);

      if (d < *min) {
        *idx = node->data.leaf.idx;
        *min = d;
      }
    }
  }

  Points const& points_;
  Metric const metric_;
  Index const dimensions_;
  internal::ItemBuffer<Node> nodes_;
  Node* root_;
};

}  // namespace nano_tree