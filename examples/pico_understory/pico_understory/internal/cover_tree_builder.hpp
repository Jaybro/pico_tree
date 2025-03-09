#pragma once

#include <cassert>
#include <numeric>
#include <random>

#include "cover_tree_base.hpp"
#include "cover_tree_data.hpp"

namespace pico_tree::internal {

template <typename SpaceWrapper_, typename Metric_, typename CoverTreeData_>
class build_cover_tree_impl {
 public:
  using index_type = typename CoverTreeData_::index_type;
  using scalar_type = typename CoverTreeData_::scalar_type;
  using node_type = typename CoverTreeData_::node_type;
  using node_allocator_type = typename CoverTreeData_::node_allocator_type;

  build_cover_tree_impl(
      SpaceWrapper_ space,
      Metric_ metric,
      scalar_type base,
      node_allocator_type& allocator)
      : space_(space), metric_(metric), base_{base}, allocator_(allocator) {}

  node_type* operator()() {
    // Both building and querying become a great deal faster for any of the
    // nearest ancestor cover trees when they are constructed with randomly
    // inserted points. It saves a huge deal of rebalancing (the build time of
    // the LiDAR dataset goes from 1.5 hour+ to about 3 minutes) and the tree
    // has a high probablity to be better balanced for queries.
    // For the simplified cover tree, query performance is greatly improved
    // using a randomized insertion at the price of construction time.
    size_t const npts = space_.size();
    std::vector<index_type> indices(npts);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    node_type* node = insert_first_two(indices);
    for (size_t i = 2; i < npts; ++i) {
      node = insert(node, create_node(indices[i]));
    }

    // Cache friendly tree.
    // TODO Take 1. A better version is probably where the contents of the
    // vector are part of the node and not in a different memory blocks.
    node_allocator_type allocator(npts);
    node_type* root = depth_first_buffer_copy(node, allocator);
    std::swap(allocator_, allocator);
    node = root;

    // TODO This is quite expensive. We can do better by using the values
    // calculated during an insert.
    // Current version is well worth it vs. queries but maybe not for high
    // dimensions.
    update_max_distance(node);

    return node;
  }

 private:
  inline node_type* create_node(index_type idx) {
    node_type* node = allocator_.allocate();
    node->index = idx;
    return node;
  }

  inline void push_child(node_type* parent, node_type* child) const {
    child->level = parent->level - scalar_type(1.0);
    parent->children.push_back(child);
  }

  inline node_type* node_to_parent(node_type* parent, node_type* child) const {
    assert(parent->is_leaf());

    parent->level = child->level + scalar_type(1.0);
    parent->children.push_back(child);
    return parent;
  }

  inline node_type* remove_leaf(node_type* tree) const {
    assert(tree->is_branch());

    node_type* node;

    do {
      node = tree;
      tree = tree->children.back();
    } while (tree->is_branch());

    node->children.pop_back();

    return tree;
  }

  //! \brief Inserts the first or first two nodes of the tree.
  //! \details Both papers don't really handle these cases but here we go.
  inline node_type* insert_first_two(std::vector<index_type> const& indices) {
    node_type* node = allocator_.allocate();
    node->index = indices[0];

    if (space_.size() == 1) {
      node->level = 0;
    } else {
      auto x0 = space_[indices[0]];
      auto x1 = space_[indices[1]];
      scalar_type d = metric_(x0, x0 + space_.sdim(), x1);
      node->level = std::ceil(base_.level(d));
      push_child(node, create_node(indices[1]));
    }

    return node;
  }

  //! \brief Returns a new tree inserting \p node into \p tree.
  inline node_type* insert(node_type* tree, node_type* node) {
    auto x = space_[node->index];
    scalar_type d = metric_(x, x + space_.sdim(), space_[tree->index]);
    scalar_type c = base_.cover_distance(*tree);
    if (d > c) {
#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
      scalar_type level = std::floor(base_.level(c + d));

      while (tree->level < level) {
        tree = node_to_parent(remove_leaf(tree), tree);
      }
#else
      c *= base_.value;
      while (d > c) {
        tree = node_to_parent(remove_leaf(tree), tree);
        c = base_.parent_distance(*tree);
        d = metric_(x, x + space_.sdim(), space_[tree->index]);
      }
#endif

      return node_to_parent(node, tree);
    } else {
      insert_covered(tree, node);
      return tree;
    }
  }

  //! \brief Insert \p node somewhere in \p tree. If the new parent for \p
  //! node already has children, rebalancing may occur.
  inline void insert_covered(node_type* tree, node_type* node) {
    // The following line may replace the contents of this function to get a
    // simplified cover tree:
    // push_child(find_parent(tree, space_[node->index], node);

    auto x = space_[node->index];
    node_type* parent = find_parent(tree, x);

    if (parent->is_leaf()) {
      push_child(parent, node);
    } else {
      rebalance(parent, node, x);
    }
  }

  template <typename PointCoords_>
  inline node_type* find_parent(node_type* tree, PointCoords_ x) const {
    if (tree->is_leaf()) {
      return tree;
    } else {
      // Traverse branches via the closest ancestors to the point. The paper
      // "Faster Cover Trees" mentions this as being part of the nearest
      // ancestor cover tree, but this speeds up the queries of the simplified
      // cover tree as well. Faster by ~70%, but tree creation is a bit slower
      // for it (3-14%).
      scalar_type min_d =
          metric_(x, x + space_.sdim(), space_[tree->children[0]->index]);
      std::size_t min_i = 0;

      for (std::size_t i = 1; i < tree->children.size(); ++i) {
        scalar_type d =
            metric_(x, x + space_.sdim(), space_[tree->children[i]->index]);

        if (d < min_d) {
          min_d = d;
          min_i = i;
        }
      }

      if (min_d <= base_.child_distance(*tree)) {
        return find_parent(tree->children[min_i], x);
      }

      return tree;
    }
  }

  template <typename PointCoords_>
  void rebalance(node_type* parent, node_type* node, PointCoords_ x) {
    std::vector<node_type*> to_move;
    std::vector<node_type*> to_stay;

    // For this node's children we want to know what the farthest distance of
    // their descendants is. That distance is this node's cover distance as it
    // is twice the separation distance. When the distance between the
    // children is more than twice this value, they don't intersect and
    // checking them can be skipped. However, since radius of this node's
    // sphere equals the cover distance, we always have to check all children.

    // TODO With the simplified nearest ancestor cover tree we can skip more
    // nodes, but perhaps we could use Node::max_distance later.

#ifdef SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE
    scalar_type const d = base_.child_distance(*parent) * scalar_type(2.0);

    for (auto child : parent->children) {
      auto y = space_[child->index];

      if (metric_(x, x + space_.sdim(), y) < d) {
        extract(x, y, child, to_move, to_stay);

        for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
          insert_covered(child, *it);
        }

        to_stay.clear();
      }
    }

    node->level = parent->level - scalar_type(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      insert_covered(node, *it);
    }

    push_child(parent, node);
#else
    for (auto& child : parent->children) {
      extract(x, space_[child->index], child, to_move, to_stay);

      for (auto it = to_stay.rbegin(); it != to_stay.rend(); ++it) {
        child = insert(child, *it);
      }

      to_stay.clear();
    }

    node->level = parent->level - scalar_type(1.0);

    for (auto it = to_move.rbegin(); it != to_move.rend(); ++it) {
      node = insert(node, *it);
    }

    push_child(parent, node);
#endif
  }

  //! \brief Fill the \p to_move and \p to_stay buffers based on if any \p
  //! descendant (of \p x_stay) is either closer to \p x_move or \p x_stay.
  //! Move is the newly inserted node for which we are rebalancing.
  template <typename PointCoords_>
  void extract(
      PointCoords_ x_move,
      PointCoords_ x_stay,
      node_type* descendant,
      std::vector<node_type*>& to_move,
      std::vector<node_type*>& to_stay) {
    if (descendant->is_leaf()) {
      return;
    }

    auto erase_begin = std::partition(
        descendant->children.begin(),
        descendant->children.end(),
        [this, &x_move, &x_stay](node_type* node) -> bool {
          auto p = space_[node->index];
          return metric_(x_stay, x_stay + space_.sdim(), p) <
                 metric_(x_move, x_move + space_.sdim(), p);
        });

    auto erase_rend =
        typename std::vector<node_type*>::reverse_iterator(erase_begin);

    auto it = descendant->children.rbegin();
    for (; it != erase_rend; ++it) {
      to_move.push_back(*it);
      strip(x_move, x_stay, *it, to_move, to_stay);
    }
    for (; it != descendant->children.rend(); ++it) {
      extract(x_move, x_stay, *it, to_move, to_stay);
    }
    descendant->children.erase(erase_begin, descendant->children.end());
  }

  //! \brief Strips all descendants of d in a depth-first fashion and puts
  //! them in either the move or stay set.
  template <typename PointCoords_>
  void strip(
      PointCoords_ x_move,
      PointCoords_ x_stay,
      node_type* descendant,
      std::vector<node_type*>& to_move,
      std::vector<node_type*>& to_stay) {
    if (descendant->is_leaf()) {
      return;
    }

    for (auto const child : descendant->children) {
      strip(x_move, x_stay, child, to_move, to_stay);

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

  node_type* depth_first_buffer_copy(
      node_type const* const node, node_allocator_type& allocator) {
    node_type* copy = allocator.allocate();
    copy->index = node->index;
    copy->level = node->level;
    copy->children.reserve(node->children.size());

    for (auto const m : node->children) {
      copy->children.push_back(depth_first_buffer_copy(m, allocator));
    }

    return copy;
  }

  void update_max_distance(node_type* node) const {
    node->max_distance = max_distance(node, space_[node->index]);

    for (node_type* m : node->children) {
      update_max_distance(m);
    }
  }

  template <typename PointCoords_>
  scalar_type max_distance(node_type const* const node, PointCoords_ x) const {
    scalar_type max = metric_(x, x + space_.sdim(), space_[node->index]);

    for (node_type const* const m : node->children) {
      max = std::max(max, max_distance(m, x));
    }

    return max;
  }

  SpaceWrapper_ space_;
  Metric_ metric_;
  base<scalar_type> base_;
  node_allocator_type& allocator_;
};

template <typename SpaceWrapper_, typename Metric_, typename Index_>
class build_cover_tree {
  using index_type = Index_;
  using scalar_type = typename SpaceWrapper_::scalar_type;
  static size_t constexpr dim = SpaceWrapper_::dim;

 public:
  using cover_tree_data_type = cover_tree_data<index_type, scalar_type>;

  //! \brief Construct a cover_tree.
  cover_tree_data_type operator()(
      SpaceWrapper_ space, Metric_ metric, scalar_type base) {
    assert(space.size() > 0);

    using build_cover_tree_impl_type =
        build_cover_tree_impl<SpaceWrapper_, Metric_, cover_tree_data_type>;
    using node_type = typename cover_tree_data_type::node_type;
    using node_allocator_type =
        typename cover_tree_data_type::node_allocator_type;

    node_allocator_type allocator(space.size());
    node_type* root_node =
        build_cover_tree_impl_type{space, metric, base, allocator}();

    return cover_tree_data_type{std::move(allocator), root_node};
  }
};

}  // namespace pico_tree::internal
