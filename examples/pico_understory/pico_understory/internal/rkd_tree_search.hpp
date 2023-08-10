#pragma once

#include <type_traits>

#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for Euclidean spaces.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class SearchNearestEuclideanDefeatist {
 public:
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  //! \brief Node type supported by this SearchNearestEuclideanDefeatist.
  using NodeType = KdTreeNodeEuclidean<IndexType, ScalarType>;

  inline SearchNearestEuclideanDefeatist(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<IndexType> const& indices,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) { SearchNearest(node); }

 private:
  inline void SearchNearest(NodeType const* const node) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        ScalarType const d =
            metric_(query_.begin(), query_.end(), space_[indices_[i]]);
        if (visitor_.max() > d) {
          visitor_(indices_[i], d);
        }
      }
    } else {
      ScalarType const v = query_[node->data.branch.split_dim];
      NodeType const* node_1st;

      // On equals we would possibly need to go left as well if we wanted exact
      // nearest neighbors. However, this search algorithm is solely used for
      // approximate nearest neighbor searches. So we'll consider it bad luck.
      // If left_max - v > 0, this means that the query is inside the left node,
      // if right_min - v < 0 it's inside the right one. For the area in between
      // we just pick the closest one by summing them.
      if ((node->data.branch.left_max + node->data.branch.right_min - v - v) >
          0) {
        node_1st = node->left;
      } else {
        node_1st = node->right;
      }

      SearchNearest(node_1st);
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<IndexType> const& indices_;
  PointWrapper_ query_;
  Visitor_& visitor_;
};

}  // namespace pico_tree::internal
