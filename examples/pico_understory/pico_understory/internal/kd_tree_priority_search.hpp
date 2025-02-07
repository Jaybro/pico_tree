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
class priority_search_nearest_euclidean {
 public:
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using point_type = point<scalar_type, SpaceWrapper_::dim>;
  //! \brief Node type supported by this priority_search_nearest_euclidean.
  using node_type = kd_tree_node_topological<index_type, scalar_type>;
  using queue_pair_type = std::pair<scalar_type, node_type const*>;

  inline priority_search_nearest_euclidean(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<index_type> const& indices,
      PointWrapper_ query,
      size_t max_leaves_visited,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        max_leaves_visited_(max_leaves_visited),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p root_node.
  inline void operator()(node_type const* const root_node) {
    std::size_t leaves_visited = 0;
    queue_.emplace(scalar_type(0.0), root_node);
    while (!queue_.empty()) {
      auto const [node_box_distance, node] = queue_.top();

      if (leaves_visited >= max_leaves_visited_ ||
          visitor_.max() < node_box_distance) {
        break;
      }

      queue_.pop();

      search_nearest(node, node_box_distance);
      ++leaves_visited;
    }
  }

 private:
  // Add nodes to the priority queue until a leaf node is reached.
  inline void search_nearest(
      node_type const* const node, scalar_type node_box_distance) {
    if (node->is_leaf()) {
      for (index_type i = node->data.leaf.begin_idx;
           i < node->data.leaf.end_idx;
           ++i) {
        visitor_(
            indices_[i],
            metric_(query_.begin(), query_.end(), space_[indices_[i]]));
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      scalar_type const v = query_[node->data.branch.split_dim];
      scalar_type old_offset;
      scalar_type new_offset;
      node_type const* node_1st;
      node_type const* node_2nd;

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
          old_offset = scalar_type(0);
        } else {
          old_offset = metric_(node->data.branch.left_min, v);
        }
        new_offset = metric_(node->data.branch.right_min, v);
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        if (v < node->data.branch.right_max) {
          old_offset = scalar_type(0);
        } else {
          old_offset = metric_(node->data.branch.right_max, v);
        }
        new_offset = metric_(node->data.branch.left_max, v);
      }

      // The distance and offset for node_1st is the same as that of its parent.
      search_nearest(node_1st, node_box_distance);

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
  std::vector<index_type> const& indices_;
  PointWrapper_ query_;
  size_t max_leaves_visited_;
  // TODO This gets created every query. Solving this would require a different
  // PicoTree interface. The queue probably shouldn't be too big.
  std::priority_queue<
      queue_pair_type,
      std::vector<queue_pair_type>,
      std::greater<queue_pair_type>>
      queue_;
  Visitor_& visitor_;
};

}  // namespace pico_tree::internal
