#pragma once

#include <queue>
#include <vector>

#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for Euclidean spaces.
//! \details S. Arya and D. M. Mount, Algorithms for fast vector quantization,
//! In IEEE Data Compression Conference, pp. 381–390, March 1993.
//! https://www.cs.umd.edu/~mount/Papers/DCC.pdf
//! This paper describes the "Priorty k-d Tree Search" technique to speed up
//! nearest neighbor queries.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class PrioritySearchNearestEuclidean {
 public:
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using PointType = Point<ScalarType, SpaceWrapper_::Dim>;
  //! \brief Node type supported by this PrioritySearchNearestEuclidean.
  using NodeType = KdTreeNodeTopological<IndexType, ScalarType>;
  using QueuePairType = std::pair<ScalarType, NodeType const*>;

  inline PrioritySearchNearestEuclidean(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<IndexType> const& indices,
      PointWrapper_ query,
      Size max_leaves_visited,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        max_leaves_visited_(max_leaves_visited),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p root_node.
  inline void operator()(NodeType const* const root_node) {
    std::size_t leaves_visited = 0;
    queue_.emplace(ScalarType(0.0), root_node);
    while (!queue_.empty()) {
      auto const [node_box_distance, node] = queue_.top();

      if (leaves_visited >= max_leaves_visited_ ||
          visitor_.max() < node_box_distance) {
        break;
      }

      queue_.pop();

      SearchNearest(node, node_box_distance);
      ++leaves_visited;
    }
  }

 private:
  // Add nodes to the priority queue until a leaf node is reached.
  inline void SearchNearest(
      NodeType const* const node, ScalarType node_box_distance) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        visitor_(
            indices_[i],
            metric_(query_.begin(), query_.end(), space_[indices_[i]]));
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      ScalarType const v = query_[node->data.branch.split_dim];
      ScalarType old_offset;
      ScalarType new_offset;
      NodeType const* node_1st;
      NodeType const* node_2nd;

      // On equals we would possibly need to go left as well. However, this is
      // handled by the if statement below this one: the check that max search
      // radius still hits the split value after having traversed the first
      // branch.
      // If left_max - v > 0, this means that the query is inside the left node,
      // if right_min - v < 0 it's inside the right one. For the area in between
      // we just pick the closest one by summing them.
      if ((node->data.branch.left_max + node->data.branch.right_min - v - v) >
          0) {
        node_1st = node->left;
        node_2nd = node->right;
        if (v > node->data.branch.left_min) {
          old_offset = ScalarType(0);
        } else {
          old_offset = metric_(node->data.branch.left_min, v);
        }
        new_offset = metric_(node->data.branch.right_min, v);
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        if (v < node->data.branch.right_max) {
          old_offset = ScalarType(0);
        } else {
          old_offset = metric_(node->data.branch.right_max, v);
        }
        new_offset = metric_(node->data.branch.left_max, v);
      }

      // The distance and offset for node_1st is the same as that of its parent.
      SearchNearest(node_1st, node_box_distance);

      // Calculate the distance to node_2nd.
      // NOTE: This method only works with Lp norms to which the exponent is not
      // applied.
      node_box_distance = node_box_distance - old_offset + new_offset;

      // Add to priority queue to be searched later.
      if (visitor_.max() > node_box_distance) {
        queue_.emplace(node_box_distance, node_2nd);
      }
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<IndexType> const& indices_;
  PointWrapper_ query_;
  Size max_leaves_visited_;
  // TODO This gets created every query. Solving this would require a different
  // PicoTree interface. The queue probably shouldn't be too big.
  std::priority_queue<
      QueuePairType,
      std::vector<QueuePairType>,
      std::greater<QueuePairType>>
      queue_;
  Visitor_& visitor_;
};

}  // namespace pico_tree::internal
