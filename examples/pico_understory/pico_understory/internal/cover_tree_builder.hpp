#pragma once

#include <cassert>
#include <numeric>
#include <random>

#include "cover_tree_base.hpp"
#include "cover_tree_data.hpp"

namespace pico_tree::internal {

template <typename SpaceWrapper_, typename Metric_, typename CoverTreeData_>
class BuildCoverTreeImpl {
 public:
  using IndexType = typename CoverTreeData_::IndexType;
  using ScalarType = typename CoverTreeData_::ScalarType;
  using NodeType = typename CoverTreeData_::NodeType;
  using NodeAllocatorType = typename CoverTreeData_::NodeAllocatorType;

  BuildCoverTreeImpl(
      SpaceWrapper_ space,
      Metric_ metric,
      ScalarType base,
      NodeAllocatorType& allocator)
      : space_(space), metric_(metric), base_{base}, allocator_(allocator) {}

  NodeType* operator()() {
    // Both building and querying become a great deal faster for any of the
    // nearest ancestor cover trees when they are constructed with randomly
    // inserted points. It saves a huge deal of rebalancing (the build time of
    // the LiDAR dataset goes from 1.5 hour+ to about 3 minutes) and the tree
    // has a high probablity to be better balanced for queries.
    // For the simplified cover tree, query performance is greatly improved
    // using a randomized insertion at the price of construction time.
    Size const npts = space_.size();
    std::vector<IndexType> indices(npts);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    NodeType* node = InsertFirstTwo(indices);
    for (IndexType i = 2; i < static_cast<IndexType>(npts); ++i) {
      node = Insert(node, CreateNode(indices[i]));
    }

    // Cache friendly tree.
    // TODO Take 1. A better version is probably where the contents of the
    // vector are part of the node and not in a different memory blocks.
    NodeAllocatorType allocator(npts);
    NodeType* root = DepthFirstBufferCopy(node, allocator);
    std::swap(allocator_, allocator);
    node = root;

    // TODO This is quite expensive. We can do better by using the values
    // calculated during an insert.
    // Current version is well worth it vs. queries but maybe not for high
    // dimensions.
    UpdateMaxDistance(node);

    return node;
  }

 private:
  inline NodeType* CreateNode(IndexType idx) {
    NodeType* node = allocator_.Allocate();
    node->index = idx;
    return node;
  }

  inline void PushChild(NodeType* parent, NodeType* child) const {
    child->level = parent->level - ScalarType(1.0);
    parent->children.push_back(child);
  }

  inline NodeType* NodeToParent(NodeType* parent, NodeType* child) const {
    assert(parent->IsLeaf());

    parent->level = child->level + ScalarType(1.0);
    parent->children.push_back(child);
    return parent;
  }

  inline NodeType* RemoveLeaf(NodeType* tree) const {
    assert(tree->IsBranch());

    NodeType* node;

    do {
      node = tree;
      tree = tree->children.back();
    } while (tree->IsBranch());

    node->children.pop_back();

    return tree;
  }

  //! \brief Inserts the first or first two nodes of the tree.
  //! \details Both papers don't really handle these cases but here we go.
  inline NodeType* InsertFirstTwo(std::vector<IndexType> const& indices) {
    NodeType* node = allocator_.Allocate();
    node->index = indices[0];

    if (space_.size() == 1) {
      node->level = 0;
    } else {
      auto x0 = space_[indices[0]];
      auto x1 = space_[indices[1]];
      ScalarType d = metric_(x0, x0 + space_.sdim(), x1);
      node->level = std::ceil(base_.Level(d));
      PushChild(node, CreateNode(indices[1]));
    }

    return node;
  }

  //! \brief Returns a new tree inserting \p node into \p tree.
  inline NodeType* Insert(NodeType* tree, NodeType* node) {
    auto x = space_[node->index];
    ScalarType d = metric_(x, x + space_.sdim(), space_[tree->index]);
    ScalarType c = base_.CoverDistance(*tree);
    if (d > c) {
#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
      ScalarType level = std::floor(base_.Level(c + d));

      while (tree->level < level) {
        tree = NodeToParent(RemoveLeaf(tree), tree);
      }
#else
      c *= base_.value;
      while (d > c) {
        tree = NodeToParent(RemoveLeaf(tree), tree);
        c = base_.ParentDistance(*tree);
        d = metric_(x, x + space_.sdim(), space_[tree->index]);
      }
#endif

      return NodeToParent(node, tree);
    } else {
      InsertCovered(tree, node);
      return tree;
    }
  }

  //! \brief Insert \p node somewhere in \p tree. If the new parent for \p
  //! node already has children, rebalancing may occur.
  inline void InsertCovered(NodeType* tree, NodeType* node) {
    // The following line may replace the contents of this function to get a
    // simplified cover tree:
    // PushChild(FindParent(tree, space_[node->index], node);

    auto x = space_[node->index];
    NodeType* parent = FindParent(tree, x);

    if (parent->IsLeaf()) {
      PushChild(parent, node);
    } else {
      Rebalance(parent, node, x);
    }
  }

  template <typename PointCoords>
  inline NodeType* FindParent(NodeType* tree, PointCoords x) const {
    if (tree->IsLeaf()) {
      return tree;
    } else {
      // Traverse branches via the closest ancestors to the point. The paper
      // "Faster Cover Trees" mentions this as being part of the nearest
      // ancestor cover tree, but this speeds up the queries of the simplified
      // cover tree as well. Faster by ~70%, but tree creation is a bit slower
      // for it (3-14%).
      ScalarType min_d =
          metric_(x, x + space_.sdim(), space_[tree->children[0]->index]);
      std::size_t min_i = 0;

      for (std::size_t i = 1; i < tree->children.size(); ++i) {
        ScalarType d =
            metric_(x, x + space_.sdim(), space_[tree->children[i]->index]);

        if (d < min_d) {
          min_d = d;
          min_i = i;
        }
      }

      if (min_d <= base_.ChildDistance(*tree)) {
        return FindParent(tree->children[min_i], x);
      }

      return tree;
    }
  }

  template <typename PointCoords>
  void Rebalance(NodeType* parent, NodeType* node, PointCoords x) {
    std::vector<NodeType*> to_move;
    std::vector<NodeType*> to_stay;

    // For this node's children we want to know what the farthest distance of
    // their descendants is. That distance is this node's cover distance as it
    // is twice the separation distance. When the distance between the
    // children is more than twice this value, they don't intersect and
    // checking them can be skipped. However, since radius of this node's
    // sphere equals the cover distance, we always have to check all children.

    // TODO With the simplified nearest ancestor cover tree we can skip more
    // nodes, but perhaps we could use Node::max_distance later.

#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
    ScalarType const d = base_.ChildDistance(*parent) * ScalarType(2.0);

    for (auto child : parent->children) {
      auto y = space_[child->index];

      if (metric_(x, x + space_.sdim(), y) < d) {
        Extract(x, y, child, to_move, to_stay);

        for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
          InsertCovered(child, *it);
        }

        to_stay.clear();
      }
    }

    node->level = parent->level - ScalarType(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      InsertCovered(node, *it);
    }

    PushChild(parent, node);
#else
    for (auto& child : parent->children) {
      Extract(x, space_[child->index], child, to_move, to_stay);

      for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
        child = Insert(child, *it);
      }

      to_stay.clear();
    }

    node->level = parent->level - ScalarType(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      node = Insert(node, *it);
    }

    PushChild(parent, node);
#endif
  }

  //! \brief Fill the \p to_move and \p to_stay buffers based on if any \p
  //! descendant (of \p x_stay) is either closer to \p x_move or \p x_stay.
  //! Move is the newly inserted node for which we are rebalancing.
  template <typename PointCoords>
  void Extract(
      PointCoords x_move,
      PointCoords x_stay,
      NodeType* descendant,
      std::vector<NodeType*>& to_move,
      std::vector<NodeType*>& to_stay) {
    if (descendant->IsLeaf()) {
      return;
    }

    auto erase_begin = std::partition(
        descendant->children.begin(),
        descendant->children.end(),
        [this, &x_move, &x_stay](NodeType* node) -> bool {
          auto p = space_[node->index];
          return metric_(x_stay, x_stay + space_.sdim(), p) <
                 metric_(x_move, x_move + space_.sdim(), p);
        });

    auto erase_rend =
        typename std::vector<NodeType*>::reverse_iterator(erase_begin);

    auto it = descendant->children.rbegin();
    for (; it != erase_rend; ++it) {
      to_move.push_back(*it);
      Strip(x_move, x_stay, *it, to_move, to_stay);
    }
    for (; it != descendant->children.rend(); ++it) {
      Extract(x_move, x_stay, *it, to_move, to_stay);
    }
    descendant->children.erase(erase_begin, descendant->children.end());
  }

  //! \brief Strips all descendants of d in a depth-first fashion and puts
  //! them in either the move or stay set.
  template <typename PointCoords>
  void Strip(
      PointCoords x_move,
      PointCoords x_stay,
      NodeType* descendant,
      std::vector<NodeType*>& to_move,
      std::vector<NodeType*>& to_stay) {
    if (descendant->IsLeaf()) {
      return;
    }

    for (auto const child : descendant->children) {
      Strip(x_move, x_stay, child, to_move, to_stay);

      auto p = space_[child->index];
      if (metric_(x_stay, x_stay + space_.sdim(), p) >
          metric_(x_move, x_move + space_.sdim(), p)) {
        to_move.push_back(child);
      } else {
        to_stay.push_back(child);
      }
    }
    descendant->children.clear();
  }

  NodeType* DepthFirstBufferCopy(
      NodeType const* const node, NodeAllocatorType& allocator) {
    NodeType* copy = allocator.Allocate();
    copy->index = node->index;
    copy->level = node->level;
    copy->children.reserve(node->children.size());

    for (auto const m : node->children) {
      copy->children.push_back(DepthFirstBufferCopy(m, allocator));
    }

    return copy;
  }

  void UpdateMaxDistance(NodeType* node) const {
    node->max_distance = MaxDistance(node, space_[node->index]);

    for (NodeType* m : node->children) {
      UpdateMaxDistance(m);
    }
  }

  template <typename PointCoords>
  ScalarType MaxDistance(NodeType const* const node, PointCoords x) const {
    ScalarType max = metric_(x, x + space_.sdim(), space_[node->index]);

    for (NodeType const* const m : node->children) {
      max = std::max(max, MaxDistance(m, x));
    }

    return max;
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  Base<ScalarType> base_;
  NodeAllocatorType& allocator_;
};

template <typename SpaceWrapper_, typename Metric_, typename Index_>
class BuildCoverTree {
  using IndexType = Index_;
  using ScalarType = typename SpaceWrapper_::ScalarType;
  static Size constexpr Dim = SpaceWrapper_::Dim;

 public:
  using CoverTreeDataType = CoverTreeData<IndexType, ScalarType>;

  //! \brief Construct a KdTree given \p points , \p max_leaf_size and
  //! SplitterType.
  CoverTreeDataType operator()(
      SpaceWrapper_ space, Metric_ metric, ScalarType base) {
    assert(space.size() > 0);

    using BuildCoverTreeImplType =
        BuildCoverTreeImpl<SpaceWrapper_, Metric_, CoverTreeDataType>;
    using NodeType = typename CoverTreeDataType::NodeType;
    using NodeAllocatorType = typename CoverTreeDataType::NodeAllocatorType;

    NodeAllocatorType allocator(space.size());
    NodeType* root_node =
        BuildCoverTreeImplType{space, metric, base, allocator}();

    return CoverTreeDataType{std::move(allocator), root_node};
  }
};

}  // namespace pico_tree::internal
