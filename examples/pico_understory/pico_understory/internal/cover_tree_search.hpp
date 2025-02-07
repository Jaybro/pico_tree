#pragma once

#include <vector>

#include "cover_tree_node.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for the cover_tree.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class search_nearest_metric {
 public:
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  using point_type = point<scalar_type, SpaceWrapper_::dim>;
  using node_type = cover_tree_node<index_type, scalar_type>;

  search_nearest_metric(
      SpaceWrapper_ space,
      Metric_ metric,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space), metric_(metric), query_(query), visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(node_type const* const node) const {
    search_nearest(node);
  }

 private:
  inline void search_nearest(node_type const* const node) const {
    scalar_type const d =
        metric_(query_.begin(), query_.end(), space_[node->index]);
    if (visitor_.max() > d) {
      visitor_(node->index, d);
    }

    std::vector<std::pair<node_type const*, scalar_type>> sorted;
    sorted.reserve(node->children.size());
    for (auto const child : node->children) {
      sorted.push_back(
          {child, metric_(query_.begin(), query_.end(), space_[child->index])});
    }

    std::sort(
        sorted.begin(),
        sorted.end(),
        [](std::pair<node_type const*, scalar_type> const& a,
           std::pair<node_type const*, scalar_type> const& b) -> bool {
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

      // TODO The distance calculation can be cached. When search_neighbor is
      // called it's calculated again.
      if (visitor_.max() >
          (metric_(query_.begin(), query_.end(), space_[m.first->index]) -
           m.first->max_distance)) {
        search_nearest(m.first);
      }
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  PointWrapper_ query_;
  Visitor_& visitor_;
};

}  // namespace pico_tree::internal
