#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! Returns a permutation based on the sorted input.
template <typename Index, typename Compare>
inline std::vector<Index> SortPermutation(Index num_points, Compare compare) {
  std::vector<Index> p(num_points);
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), std::move(compare));
  return p;
}

//! Compile time dimension information relative to Dim.
template <int Dim>
struct Dimension {
  //! Compile time determination of the next dimension.
  inline static constexpr int Next() { return Dim + 1; }
  //! Return the current "x" dimension relative to Dim.
  inline static constexpr int d0(int) { return Dim; }
  //! Return the current "y" dimension relative to Dim.
  inline static constexpr int d1(int) { return Dim + 1; }
};

//! Run time dimension information relative to some input dimension.
template <>
struct Dimension<kRuntimeDims> {
  //! Compile time determination of the next dimension. Unknown in case of
  //! runtime based tree creation.
  inline static constexpr int Next() { return kRuntimeDims; }
  //! Return the current "x" dimension relative to \p dim .
  inline static int d0(int dim) { return dim; }
  //! Return the current "y" dimension relative to \p dim .
  inline static int d1(int dim) { return dim + 1; }
};

//! Node represents a tree node that is meant to be inherited.
template <typename Derived, typename Index, typename Scalar>
struct NodeBase {
  union Data {
    struct Branch {
      Scalar split;
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
  Derived* left;
  Derived* right;
};

//! The RangeLayer is the last dimension in any 2+ dimensional range tree to
//! improve query time by a factor of O(log_2 n) using fractional cascading.
//! https://en.wikipedia.org/wiki/Fractional_cascading
//! Each RangeLayer corresponds to a single sub-tree / array that contains
//! cascade information using the Item class.
template <typename Index, typename Scalar, int Dim, typename Points>
class RangeLayer {
 public:
  struct Item {
    //! Returns the index range represented by this Item.
    inline std::pair<Index, Index> range() const {
      return std::make_pair(left, right);
    }
    //! Point index of the input point set.
    Index index;
    //! Item index in the left sub RangeLayer. The item on this index is the
    //! first index with a dimensional element value that is higher or equal to
    //! the one represented by this Item. Equals the size of the array in case
    //! no such value exists.
    Index left;
    //! Same as left, but then for the right sub RangeLayer.
    Index right;
  };

  RangeLayer(Points const& points, Index const dimension, Index num_points)
      : points_{points},
        dimension_(dimension),
        items_{static_cast<std::size_t>(num_points)} {}

  inline Index LowerBound(Scalar value) const {
    return std::lower_bound(
               items_.cbegin(),
               items_.cend(),
               value,
               [this](Item const& a, Scalar const b) {
                 return operator()(a) < b;
               }) -
           items_.cbegin();
  }

  inline Index UpperBound(Index lower_bound, Scalar value) const {
    return std::upper_bound(
               items_.cbegin() + lower_bound,
               items_.cend(),
               value,
               [this](Scalar const a, Item const& b) {
                 return a < operator()(b);
               }) -
           items_.cbegin();
  }

  inline std::vector<Item> const& data() const { return items_; }
  inline std::vector<Item>& data() { return items_; }
  //! Returns the range at Item index \p i.
  inline std::pair<Index, Index> range(Index i) const {
    return items_[i].range();
  }
  //! Returns the range at Item index \p i. In case the index is out of bounds,
  //! a range is created having values \p l and \p r .
  inline std::pair<Index, Index> range(Index i, Index l, Index r) const {
    if (i < items_.size()) {
      return items_[i].range();
    } else {
      return std::make_pair(l, r);
    }
  }
  inline Index size() const { return items_.size(); }

 private:
  //! Returns the value of a point having sorted index \p i .
  inline Scalar operator()(Item i) const {
    return points_(i.index, Dimension<Dim>::d0(dimension_));
  }

  Points const& points_;
  int dimension_;
  std::vector<Item> items_;
};

template <typename Index, typename Scalar, int Dim, typename Points>
class RangeTree2d_ {
 private:
  using Layer = RangeLayer<Index, Scalar, Dimension<Dim>::Next(), Points>;
  using Item = typename Layer::Item;

  struct Node : public NodeBase<Node, Index, Scalar> {
    Layer* layer;
  };

 public:
  explicit RangeTree2d_(Points const& points)
      : points_{points},
        dimension_{0},
        nodes_{MaxNodesFromPoints(points_.num_points())},
        layers_{MaxNodesFromPoints(points_.num_points())},
        root_{MakeTree()} {
    assert(points_.num_points() > 0);
  }

  //! Constructs a 2d tree from the last two dimensions of a higher dimensional
  //! tree.
  //! \details Since we may be building a tree for any of the higher dimensional
  //! sub-trees, the size of this tree can't be determined from the size of the
  //! point set. Instead, we the size of \p direct_p_by_x.
  //! Note that the x and y dimensions are considered relative to \p dimension .
  //! \param points Point set used for building the tree.
  //! \param dimension First dimension of this 2d tree.
  //! \param direct_p_by_x Sorted indices for the first dimension.
  //! \param buffer Memory buffer of minimally direct_p_by_x.size() elments.
  RangeTree2d_(
      Points const& points,
      Index const dimension,
      std::vector<Index> const& direct_p_by_x,
      std::vector<Index>* buffer)
      : points_{points},
        dimension_{dimension},
        nodes_{MaxNodesFromPoints(direct_p_by_x.size())},
        layers_{MaxNodesFromPoints(direct_p_by_x.size())},
        root_{MakeTree(direct_p_by_x, buffer)} {
    assert(direct_p_by_x.size() > 0);
  }

  //! Perform a range search in O(log_2 n) time.
  template <typename P>
  inline void SearchRange(
      P const& min, P const& max, std::vector<Index>* indices) const {
    assert(
        points_(min, Dimension<Dim>::d0(dimension_)) <=
        points_(max, Dimension<Dim>::d0(dimension_)));
    assert(
        points_(min, Dimension<Dim>::d1(dimension_)) <=
        points_(max, Dimension<Dim>::d1(dimension_)));

    // Find the first node where min "x" and max "x" split.
    Node* split = root_;
    if (split->IsBranch()) {
      do {
        if (points_(max, Dimension<Dim>::d0(dimension_)) <
            split->data.branch.split) {
          do {
            split = split->left;
          } while (split->IsBranch() &&
                   points_(max, Dimension<Dim>::d0(dimension_)) <
                       split->data.branch.split);
        } else if (
            points_(min, Dimension<Dim>::d0(dimension_)) >
            split->data.branch.split) {
          do {
            split = split->right;
          } while (split->IsBranch() &&
                   points_(min, Dimension<Dim>::d0(dimension_)) >
                       split->data.branch.split);
        } else {
          break;
        }
      } while (split->IsBranch());

      // If we split at a branch we traverse the left and right sub trees to
      // report their contents.
      // On the left side of the split we report all right sub-trees and on the
      // right side of the split we report all left sub-trees.
      // At the same time we find the correspoding "y" range in the "y"
      // dimension. As we go down in the "x" dimension the "y" dimension follows
      // the same branches using the RangeLayer cascade.
      if (split->IsBranch()) {
        auto lower_bound = split->layer->LowerBound(
            points_(min, Dimension<Dim>::d1(dimension_)));
        // Check if there is any data at all.
        if (lower_bound < split->layer->size()) {
          auto upper_bound = split->layer->UpperBound(
              lower_bound, points_(max, Dimension<Dim>::d1(dimension_)));
          std::pair<Index, Index> c_lower = split->layer->range(lower_bound);
          std::pair<Index, Index> c_upper = split->layer->range(
              upper_bound,
              split->left->layer->size(),
              split->right->layer->size());

          // Left side of the split branch.
          Node* track = split->left;
          lower_bound = c_lower.first;
          upper_bound = c_upper.first;
          while (track->IsBranch() && lower_bound < track->layer->size()) {
            std::pair<Index, Index> track_c_lower =
                track->layer->range(lower_bound);
            std::pair<Index, Index> track_c_upper = track->layer->range(
                upper_bound,
                track->left->layer->size(),
                track->right->layer->size());

            if (points_(min, Dimension<Dim>::d0(dimension_)) <=
                track->data.branch.split) {
              ReportIndices(
                  track->right->layer->data(),
                  track_c_lower.second,
                  track_c_upper.second,
                  indices);

              track = track->left;
              lower_bound = track_c_lower.first;
              upper_bound = track_c_upper.first;
            } else {
              track = track->right;
              lower_bound = track_c_lower.second;
              upper_bound = track_c_upper.second;
            }
          }

          // Last leaf of the left side.
          if (track->IsLeaf() &&
              points_(min, Dimension<Dim>::d0(dimension_)) <=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
              points_(min, Dimension<Dim>::d1(dimension_)) <=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d1(dimension_)) &&
              points_(max, Dimension<Dim>::d1(dimension_)) >=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d1(dimension_))) {
            indices->push_back(track->data.leaf.index);
          }

          // Right side of the split branch.
          track = split->right;
          lower_bound = c_lower.second;
          upper_bound = c_upper.second;
          while (track->IsBranch() && lower_bound < track->layer->size()) {
            std::pair<Index, Index> track_c_lower =
                track->layer->range(lower_bound);
            std::pair<Index, Index> track_c_upper = track->layer->range(
                upper_bound,
                track->left->layer->size(),
                track->right->layer->size());

            if (points_(max, Dimension<Dim>::d0(dimension_)) >=
                track->data.branch.split) {
              ReportIndices(
                  track->left->layer->data(),
                  track_c_lower.first,
                  track_c_upper.first,
                  indices);

              track = track->right;
              lower_bound = track_c_lower.second;
              upper_bound = track_c_upper.second;
            } else {
              track = track->left;
              lower_bound = track_c_lower.first;
              upper_bound = track_c_upper.first;
            }
          }

          // Last leaf of the right side.
          if (track->IsLeaf() &&
              points_(max, Dimension<Dim>::d0(dimension_)) >=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
              points_(min, Dimension<Dim>::d1(dimension_)) <=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d1(dimension_)) &&
              points_(max, Dimension<Dim>::d1(dimension_)) >=
                  points_(
                      track->data.leaf.index, Dimension<Dim>::d1(dimension_))) {
            indices->push_back(track->data.leaf.index);
          }
        }
      } else {
        // We never found a split node and ended up in a leaf.
        if (points_(min, Dimension<Dim>::d0(dimension_)) <=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
            points_(min, Dimension<Dim>::d1(dimension_)) <=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d1(dimension_)) &&
            points_(max, Dimension<Dim>::d0(dimension_)) >=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
            points_(max, Dimension<Dim>::d1(dimension_)) >=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d1(dimension_))) {
          indices->push_back(split->data.leaf.index);
        }
      }
    } else {
      // The root is a leaf.
      if (points_(min, Dimension<Dim>::d0(dimension_)) <=
              points_(split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
          points_(min, Dimension<Dim>::d1(dimension_)) <=
              points_(split->data.leaf.index, Dimension<Dim>::d1(dimension_)) &&
          points_(max, Dimension<Dim>::d0(dimension_)) >=
              points_(split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
          points_(max, Dimension<Dim>::d1(dimension_)) >=
              points_(split->data.leaf.index, Dimension<Dim>::d1(dimension_))) {
        indices->push_back(split->data.leaf.index);
      }
    }
  }

  inline Points const& points() const { return points_; }

 private:
  //! Builds the tree in O(3 * n log_2 n) time.
  inline Node* MakeTree() {
    auto const& points = points_;
    Index const dimension = dimension_;

    // Sorted x indices that directly refer to a point.
    std::vector<Index> direct_p_by_x{SortPermutation(
        points.num_points(), [&points, dimension](Index i, Index j) -> bool {
          return points(i, Dimension<Dim>::d0(dimension)) <
                 points(j, Dimension<Dim>::d0(dimension));
        })};

    std::vector<Index> buffer(direct_p_by_x.size());
    return MakeTree(direct_p_by_x, &buffer);
  }

  //! Builds the tree in O(2 * n log_2 n) time using pre-sorted indices of the
  //! first dimension.
  inline Node* MakeTree(
      std::vector<Index> const& direct_p_by_x, std::vector<Index>* p_buffer) {
    auto const& points = points_;
    Index const dimension = dimension_;
    Index const num_points = direct_p_by_x.size();

    // Sorted y indices that link to the points via the x dimension.
    std::vector<Index> linked_x_by_y{SortPermutation(
        num_points,
        [&points, dimension, &direct_p_by_x](Index i, Index j) -> bool {
          return points(direct_p_by_x[i], Dimension<Dim>::d1(dimension)) <
                 points(direct_p_by_x[j], Dimension<Dim>::d1(dimension));
        })};

    // Root associate layer.
    Layer* parent =
        layers_.MakeItem(points_, Dimension<Dim>::d1(dimension_), num_points);
    for (Index i = 0; i < num_points; ++i) {
      parent->data()[i].index = direct_p_by_x[linked_x_by_y[i]];
    }

    return SplitIndices(
        direct_p_by_x, num_points / 2, parent, &linked_x_by_y, p_buffer);
  }

  inline Node* SplitIndices(
      std::vector<Index> const& direct_p_by_x,
      Index const split,
      Layer* parent,
      std::vector<Index>* p_front,
      std::vector<Index>* p_back) {
    // Leaf
    if (parent->data().size() == 1) {
      Node* node = nodes_.MakeItem();
      node->layer = parent;
      node->data.leaf.index = direct_p_by_x[split];
      node->left = nullptr;
      node->right = nullptr;
      return node;
    }

    std::vector<Index> const& front = *p_front;
    std::vector<Index>& back = *p_back;

    // Right may be one bigger than left
    Index const parent_size = parent->data().size();
    Index const left_size = parent_size / 2;
    Index const right_size = parent_size - left_size;
    Index const left_offset = split - left_size;
    Index const right_offset = split;
    Layer* left =
        layers_.MakeItem(points_, Dimension<Dim>::d1(dimension_), left_size);
    Layer* right =
        layers_.MakeItem(points_, Dimension<Dim>::d1(dimension_), right_size);
    Index left_count = 0;
    Index right_count = 0;

    // Cascade to the next layer.
    for (Index i = 0; i < parent_size; ++i) {
      // Index in the next layer to the smallest value which is greater or
      // equal to the current one. Since the sequence is ordened, the current
      // value to be inserted must be the smallest one. The index will be the
      // next to be inserted.
      // TODO Test this later =D
      // NOTE: In case there is an equals range, we should check the last
      // existing entry for equality. Practically, this gains us nothing as we
      // should always arrive at the start of the equals range (or the end).
      parent->data()[i].left = left_count;
      parent->data()[i].right = right_count;
      // We move the sorted y values into the left or right node based on the
      // sorted x indices while keeping their relative ordering.
      // Split is the first index on the right side.
      Index const index_prev_dim = front[left_offset + i];
      if (index_prev_dim < right_offset) {
        left->data()[left_count].index = parent->data()[i].index;
        back[left_offset + left_count] = index_prev_dim;
        ++left_count;
      } else {
        right->data()[right_count].index = parent->data()[i].index;
        back[right_offset + right_count] = index_prev_dim;
        ++right_count;
      }
    }

    Node* node = nodes_.MakeItem();
    node->layer = parent;
    node->data.branch.split =
        points_(direct_p_by_x[split], Dimension<Dim>::d0(dimension_));
    node->left = SplitIndices(
        direct_p_by_x, left_offset + left_size / 2, left, p_back, p_front);
    node->right = SplitIndices(
        direct_p_by_x, right_offset + right_size / 2, right, p_back, p_front);
    return node;
  }

  inline void ReportIndices(
      std::vector<Item> const& items,
      Index const lower_bound,
      Index const upper_bound,
      std::vector<Index>* indices) const {
    // For MingW GCC 9.2.0, this was hands down the fastest for
    // copying indices. Vs. for_each, or any other type of for/while loop.
    std::transform(
        items.cbegin() + lower_bound,
        items.cbegin() + upper_bound,
        std::back_inserter(*indices),
        [](Item const& i) { return i.index; });
  }

  Points const& points_;
  Index dimension_;
  StaticBuffer<Node> nodes_;
  StaticBuffer<Layer> layers_;
  Node* root_;
};

template <typename Index, typename Scalar, int Dim, typename Points>
class RangeTreeNd_ {
 private:
  // TODO Hardcoded 3 dimensions
  // TODO However, it should perform better first before generalizing.
  static constexpr int Dims = 3;

  using RangeTree2d =
      RangeTree2d_<Index, Scalar, Dimensions<Dims>::Back(2), Points>;
  using RangeTreeNd = RangeTreeNd_<Index, Scalar, Dim, Points>;

  struct Node : public NodeBase<Node, Index, Scalar> {
    union Associate {
      RangeTreeNd* nd;
      RangeTree2d* td;
    };
    Associate sub;
  };

 public:
  explicit RangeTreeNd_(Points const& points)
      : points_{points},
        dimension_{0},
        nodes_{MaxNodesFromPoints(points_.num_points())},
        trees_nd_{0},
        trees_2d_{MaxNodesFromPoints(points_.num_points())},
        root_{MakeTree()} {
    assert(points_.num_points() > 0);
  }

  //! Perform range search in O(log_2^max(1, d - 1) n) time.
  template <typename P>
  inline void SearchRange(
      P const& min, P const& max, std::vector<Index>* indices) const {
    assert(
        points_(min, Dimension<Dim>::d0(dimension_)) <=
        points_(max, Dimension<Dim>::d0(dimension_)));
    // Find node where min_x and max_x split.
    Node* split = root_;
    if (split->IsBranch()) {
      do {
        if (points_(max, Dimension<Dim>::d0(dimension_)) <
            split->data.branch.split) {
          do {
            split = split->left;
          } while (split->IsBranch() &&
                   points_(max, Dimension<Dim>::d0(dimension_)) <
                       split->data.branch.split);
        } else if (
            points_(min, Dimension<Dim>::d0(dimension_)) >
            split->data.branch.split) {
          do {
            split = split->right;
          } while (split->IsBranch() &&
                   points_(min, Dimension<Dim>::d0(dimension_)) >
                       split->data.branch.split);
        } else {
          break;
        }
      } while (split->IsBranch());

      if (split->IsBranch()) {
        // Left side of the split branch.
        Node* track = split->left;
        while (track->IsBranch()) {
          if (points_(min, Dimension<Dim>::d0(dimension_)) <=
              track->data.branch.split) {
            track->right->sub.td->SearchRange(min, max, indices);
            track = track->left;
          } else {
            track = track->right;
          }
        }

        // Last leaf of the left side.
        if (points_(min, Dimension<Dim>::d0(dimension_)) <=
            points_(track->data.leaf.index, Dimension<Dim>::d0(dimension_))) {
          track->sub.td->SearchRange(min, max, indices);
        }

        // Right side of the split branch.
        track = split->right;
        while (track->IsBranch()) {
          if (points_(max, Dimension<Dim>::d0(dimension_)) >=
              track->data.branch.split) {
            track->left->sub.td->SearchRange(min, max, indices);
            track = track->right;
          } else {
            track = track->left;
          }
        }

        // Last leaf of the right side.
        if (points_(max, Dimension<Dim>::d0(dimension_)) >=
            points_(track->data.leaf.index, Dimension<Dim>::d0(dimension_))) {
          track->sub.td->SearchRange(min, max, indices);
        }
      } else {
        // We never found a split node and ended up in a leaf.
        if (points_(min, Dimension<Dim>::d0(dimension_)) <=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
            points_(max, Dimension<Dim>::d0(dimension_)) >=
                points_(
                    split->data.leaf.index, Dimension<Dim>::d0(dimension_))) {
          split->sub.td->SearchRange(min, max, indices);
        }
      }
    } else {
      // The root is a leaf.
      if (points_(min, Dimension<Dim>::d0(dimension_)) <=
              points_(split->data.leaf.index, Dimension<Dim>::d0(dimension_)) &&
          points_(max, Dimension<Dim>::d0(dimension_)) >=
              points_(split->data.leaf.index, Dimension<Dim>::d0(dimension_))) {
        split->sub.td->SearchRange(min, max, indices);
      }
    }
  }

  inline Points const& points() const { return points_; }

 private:
  inline Node* MakeTree() {
    auto const& points = points_;

    // Sorted x indices that directly refer to a point.
    std::vector<Index> direct_p_by_x{SortPermutation(
        points.num_points(), [&points](Index i, Index j) -> bool {
          return points(i, 0) < points(j, 0);
        })};

    // Sorted y indices that link to the points via the x dimension.
    std::vector<Index> linked_x_by_y{SortPermutation(
        points.num_points(),
        [&points, &direct_p_by_x](Index i, Index j) -> bool {
          return points(direct_p_by_x[i], 1) < points(direct_p_by_x[j], 1);
        })};

    std::vector<Index> parent(points.num_points());
    for (Index i = 0; i < points.num_points(); ++i) {
      parent[i] = direct_p_by_x[linked_x_by_y[i]];
    }

    std::vector<Index> buffer(points.num_points());
    return SplitIndices(
        direct_p_by_x,
        points.num_points() / 2,
        std::move(parent),
        &linked_x_by_y,
        &buffer);
  }

  inline Node* SplitIndices(
      std::vector<Index> const& direct_p_by_d,
      Index const split,
      std::vector<Index>&& parent,
      std::vector<Index>* p_front,
      std::vector<Index>* p_back) {
    Node* node = nodes_.MakeItem();
    node->sub.td = trees_2d_.MakeItem(
        points_, Dimension<Dim>::d1(dimension_), parent, p_back);

    // Leaf
    if (parent.size() == 1) {
      node->data.leaf.index = direct_p_by_d[split];
      node->left = nullptr;
      node->right = nullptr;
    } else {
      std::vector<Index> const& front = *p_front;
      std::vector<Index>& back = *p_back;

      // Right may be one bigger than right
      Index const parent_size = parent.size();
      Index const left_size = parent_size / 2;
      Index const right_size = parent_size - left_size;
      Index const left_offset = split - left_size;
      Index const right_offset = split;
      std::vector<Index> left(left_size);
      std::vector<Index> right(right_size);
      Index left_count = 0;
      Index right_count = 0;

      // Split indices.
      for (Index i = 0; i < parent_size; ++i) {
        Index const index_prev_dim = front[left_offset + i];
        if (index_prev_dim < right_offset) {
          left[left_count] = parent[i];
          back[left_offset + left_count] = index_prev_dim;
          ++left_count;
        } else {
          right[right_count] = parent[i];
          back[right_offset + right_count] = index_prev_dim;
          ++right_count;
        }
      }

      node->data.branch.split =
          points_(direct_p_by_d[split], Dimension<Dim>::d0(dimension_));
      node->left = SplitIndices(
          direct_p_by_d,
          left_offset + left_size / 2,
          std::move(left),
          p_back,
          p_front);
      node->right = SplitIndices(
          direct_p_by_d,
          right_offset + right_size / 2,
          std::move(right),
          p_back,
          p_front);
    }

    return node;
  }

  Points const& points_;
  Index dimension_;
  StaticBuffer<Node> nodes_;
  StaticBuffer<RangeTreeNd> trees_nd_;
  StaticBuffer<RangeTree2d> trees_2d_;
  Node* root_;
};

}  // namespace internal

template <typename Index, typename Scalar, typename Points>
class RangeTree1d {
 public:
  explicit RangeTree1d(Points const& points)
      : points_{points},
        sorted_{internal::SortPermutation(
            points_.num_points(), [this](int i, int j) -> bool {
              return operator()(i) < operator()(j);
            })} {
    assert(points_.num_points() > 0);
  }

  //! Back-inserts the indices into \p indices representing all points
  //! within the range [ \p min ... \p max ].
  inline void SearchRange(
      Scalar min, Scalar max, std::vector<Index>* indices) const {
    auto it_lower = std::lower_bound(
        sorted_.cbegin(), sorted_.cend(), min, [this](Index a, Scalar b) {
          return operator()(a) < b;
        });

    auto it_upper = std::upper_bound(
        it_lower, sorted_.cend(), max, [this](Scalar a, Index b) {
          return a < operator()(b);
        });

    std::copy(it_lower, it_upper, std::back_inserter(*indices));
  }

  //! Returns the index of a point closest to \p value .
  inline Index SearchNearest(Scalar const& value) const {
    auto it = std::lower_bound(
        sorted_.cbegin(), sorted_.cend(), value, [this](Index a, Scalar b) {
          return operator()(a) < b;
        });

    if (it != sorted_.begin()) {
      auto prev = std::prev(it);
      if ((operator()(*it) - value) > (value - operator()(*prev))) {
        return *prev;
      }
    }

    return *it;
  }

  inline Points const& points() const { return points_; }

 private:
  //! Returns the value of a point having sorted index \p i .
  inline Scalar operator()(Index i) const { return points_(i, 0); }

  Points const& points_;
  std::vector<Index> sorted_;
};

template <typename Index, typename Scalar, typename Points>
class RangeTree2d : public internal::RangeTree2d_<Index, Scalar, 0, Points> {
 public:
  explicit RangeTree2d(Points const& points)
      : internal::RangeTree2d_<Index, Scalar, 0, Points>(points) {}
};

}  // namespace pico_tree
