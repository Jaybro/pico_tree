#pragma once

#include <vector>

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_node.hpp"
#include "pico_tree/internal/point.hpp"
#include "pico_tree/metric.hpp"

namespace pico_tree::internal {

//! \brief This class provides a search nearest function for Euclidean spaces.
//! \details S. Arya and D. M. Mount, Algorithms for fast vector quantization,
//! In IEEE Data Compression Conference, pp. 381–390, March 1993.
//! https://www.cs.umd.edu/~mount/Papers/DCC.pdf
//! This paper describes the "Incremental Distance Calculation" technique  to
//! speed up nearest neighbor queries.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class SearchNearestEuclidean {
 public:
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using PointType = Point<ScalarType, SpaceWrapper_::Dim>;
  //! \brief Node type supported by this SearchNearestEuclidean.
  using NodeType = KdTreeNodeEuclidean<IndexType, ScalarType>;

  inline SearchNearestEuclidean(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<IndexType> const& indices,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(PointType::FromSize(space_.sdim())),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) {
    node_box_offset_.Fill(ScalarType(0.0));
    SearchNearest(node, ScalarType(0.0));
  }

 private:
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
        new_offset = metric_(node->data.branch.right_min, v);
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = metric_(node->data.branch.left_max, v);
      }

      // The distance and offset for node_1st is the same as that of its parent.
      SearchNearest(node_1st, node_box_distance);

      // Calculate the distance to node_2nd.
      // NOTE: This method only works with Lp norms to which the exponent is not
      // applied.
      ScalarType const old_offset =
          node_box_offset_[node->data.branch.split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[node->data.branch.split_dim] = new_offset;
        SearchNearest(node_2nd, node_box_distance);
        node_box_offset_[node->data.branch.split_dim] = old_offset;
      }
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<IndexType> const& indices_;
  PointWrapper_ query_;
  PointType node_box_offset_;
  Visitor_& visitor_;
};

//! \brief This class provides a search nearest function for topological spaces.
template <
    typename SpaceWrapper_,
    typename Metric_,
    typename PointWrapper_,
    typename Visitor_,
    typename Index_>
class SearchNearestTopological {
 public:
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  using PointType = Point<ScalarType, SpaceWrapper_::Dim>;
  //! \brief Node type supported by this SearchNearestTopological.
  using NodeType = KdTreeNodeTopological<IndexType, ScalarType>;

  inline SearchNearestTopological(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<IndexType> const& indices,
      PointWrapper_ query,
      Visitor_& visitor)
      : space_(space),
        metric_(metric),
        indices_(indices),
        query_(query),
        node_box_offset_(PointType::FromSize(space_.sdim())),
        visitor_(visitor) {}

  //! \brief Search nearest neighbors starting from \p node.
  inline void operator()(NodeType const* const node) {
    node_box_offset_.Fill(ScalarType(0.0));
    SearchNearest(node, ScalarType(0.0));
  }

 private:
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
      // Determine the distance to the boxes of the children of this node.
      ScalarType const d1 = metric_(
          v,
          node->data.branch.left_min,
          node->data.branch.left_max,
          node->data.branch.split_dim);
      ScalarType const d2 = metric_(
          v,
          node->data.branch.right_min,
          node->data.branch.right_max,
          node->data.branch.split_dim);
      NodeType const* node_1st;
      NodeType const* node_2nd;
      ScalarType new_offset;

      // Visit the closest child/box first.
      if (d1 < d2) {
        node_1st = node->left;
        node_2nd = node->right;
        new_offset = d2;
      } else {
        node_1st = node->right;
        node_2nd = node->left;
        new_offset = d1;
      }

      SearchNearest(node_1st, node_box_distance);

      ScalarType const old_offset =
          node_box_offset_[node->data.branch.split_dim];
      node_box_distance = node_box_distance - old_offset + new_offset;

      // The value visitor->max() contains the current nearest neighbor distance
      // or otherwise current maximum search distance. When testing against the
      // split value we determine if we should go into the neighboring node.
      if (visitor_.max() >= node_box_distance) {
        node_box_offset_[node->data.branch.split_dim] = new_offset;
        SearchNearest(node_2nd, node_box_distance);
        node_box_offset_[node->data.branch.split_dim] = old_offset;
      }
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<IndexType> const& indices_;
  PointWrapper_ query_;
  PointType node_box_offset_;
  Visitor_& visitor_;
};

//! \brief A functor that provides range searches for Euclidean spaces. Query
//! time is bounded by O(n^(1-1/Dim)+k).
//! \details Many tree nodes are excluded by checking if they intersect with the
//! box of the query. We don't store the bounding box of each node but calculate
//! them at run time. This slows down SearchBox in favor of having faster
//! nearest neighbor searches.
template <typename SpaceWrapper_, typename Metric_, typename Index_>
class SearchBoxEuclidean {
 public:
  // TODO Perhaps we can support it for both topological and Euclidean spaces.
  static_assert(
      std::is_same_v<typename Metric_::SpaceTag, EuclideanSpaceTag>,
      "SEARCH_BOX_ONLY_SUPPORTED_FOR_EUCLIDEAN_SPACES");

  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  static Size constexpr Dim = SpaceWrapper_::Dim;
  using BoxType = Box<ScalarType, Dim>;
  using BoxMapType = BoxMap<ScalarType const, Dim>;

  inline SearchBoxEuclidean(
      SpaceWrapper_ space,
      Metric_ metric,
      std::vector<IndexType> const& indices,
      BoxType const& root_box,
      BoxMapType const& query,
      std::vector<IndexType>& idxs)
      : space_(space),
        metric_(metric),
        indices_(indices),
        box_(root_box),
        query_(query),
        idxs_(idxs) {}

  //! \brief Range search starting from \p node.
  template <typename Node>
  inline void operator()(Node const* const node) {
    if (node->IsLeaf()) {
      for (IndexType i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        if (query_.Contains(space_[indices_[i]])) {
          idxs_.push_back(indices_[i]);
        }
      }
    } else {
      ScalarType old_value = box_.max(node->data.branch.split_dim);
      box_.max(node->data.branch.split_dim) = node->data.branch.left_max;

      // Check if the left node is fully contained. If true, report all its
      // indices. Else, if its partially contained, continue the range search
      // down the left node.
      if (query_.Contains(box_)) {
        ReportNode(node->left);
      } else if (
          query_.min(node->data.branch.split_dim) <
          node->data.branch.left_max) {
        operator()(node->left);
      }

      box_.max(node->data.branch.split_dim) = old_value;
      old_value = box_.min(node->data.branch.split_dim);
      box_.min(node->data.branch.split_dim) = node->data.branch.right_min;

      // Same as the left side.
      if (query_.Contains(box_)) {
        ReportNode(node->right);
      } else if (
          query_.max(node->data.branch.split_dim) >
          node->data.branch.right_min) {
        operator()(node->right);
      }

      box_.min(node->data.branch.split_dim) = old_value;
    }
  }

 private:
  //! \brief Reports all indices contained by \p node.
  template <typename Node>
  inline void ReportNode(Node const* const node) const {
    IndexType begin;
    IndexType end;

    if (node->IsLeaf()) {
      begin = node->data.leaf.begin_idx;
      end = node->data.leaf.end_idx;
    } else {
      // Nodes and index pointers (begin_idx and end_idx) are ordered left to
      // right. This means that for any node, its left-most and right-most leaf
      // node descendants will respectively store the begin index and end index
      // of the entire range of points contained by that node.
      begin = ReportLeft(node->left);
      end = ReportRight(node->right);
    }

    std::copy(
        indices_.cbegin() + begin,
        indices_.cbegin() + end,
        std::back_inserter(idxs_));
  }

  template <typename Node>
  inline IndexType ReportLeft(Node const* const node) const {
    if (node->IsLeaf()) {
      return node->data.leaf.begin_idx;
    } else {
      return ReportLeft(node->left);
    }
  }

  template <typename Node>
  inline IndexType ReportRight(Node const* const node) const {
    if (node->IsLeaf()) {
      return node->data.leaf.end_idx;
    } else {
      return ReportRight(node->right);
    }
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  std::vector<IndexType> const& indices_;
  // This variable is used for maintaining a running bounding box.
  BoxType box_;
  BoxMapType const& query_;
  std::vector<IndexType>& idxs_;
};

}  // namespace pico_tree::internal
