#pragma once

#include <vector>

#include "cover_tree_node.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for the CoverTree.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class SearchNearestMetric {
 public:
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using PointType = Point<ScalarType, SpaceWrapper_::Dim>;
  using NodeType = CoverTreeNode<IndexType, ScalarType>;

  SearchNearestMetric(
      SpaceWrapper_ const& space,
      Metric_ const& metric,
      PointWrapper_ const& query,
      Visitor_& visitor)
      : space_(space), metric_(metric), query_(query), visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) const {
    SearchNearest(node);
  }

 private:
  inline void SearchNearest(NodeType const* const node) const {
    ScalarType const d =
        metric_(query_.begin(), query_.end(), space_[node->index]);
    if (visitor_.max() > d) {
      visitor_(node->index, d);
    }

    std::vector<std::pair<NodeType const*, ScalarType>> sorted;
    sorted.reserve(node->children.size());
    for (auto const child : node->children) {
      sorted.push_back(
          {child, metric_(query_.begin(), query_.end(), space_[child->index])});
    }

    std::sort(
        sorted.begin(),
        sorted.end(),
        [](std::pair<NodeType const*, ScalarType> const& a,
           std::pair<NodeType const*, ScalarType> const& b) -> bool {
          return a.second < b.second;
        });

    for (auto const& m : sorted) {
      // Algorithm 1 from paper "Faster Cover Trees" has a mistake. It checks
      // with respect to the nearest point, not the query point itself,
      // intersecting the wrong spheres.
      // Algorithm 1 from paper "Cover Trees for Nearest Neighbor" is correct.

      // The upper-bound distance a descendant can be is twice the cover
      // distance of the node. This is true taking the invariants into account.
      // NOTE: In "Cover Trees for Nearest Neighbor" this upper-bound
      // practically appears to be half this distance, as new nodes are only
      // added when they are within cover distance.
      // For "Faster Cover Trees" it is twice the cover distance due to the
      // first phase of the insert algorithm (not having a root at infinity).

      // TODO The distance calculation can be cached. When SearchNeighbor is
      // called it's calculated again.
      if (visitor_.max() >
          (metric_(query_.begin(), query_.end(), space_[m.first->index]) -
           m.first->max_distance)) {
        SearchNearest(m.first);
      }
    }
  }

  SpaceWrapper_ const& space_;
  Metric_ const& metric_;
  PointWrapper_ const& query_;
  Visitor_& visitor_;
};

}  // namespace pico_tree::internal
