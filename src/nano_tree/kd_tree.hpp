#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include "core.hpp"

namespace nano_tree {

namespace internal {

template <typename Index, typename Scalar>
class SearchNn {
 public:
  SearchNn() : min_{std::numeric_limits<Scalar>::max()} {}

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    if (d < min_) {
      idx_ = idx;
      min_ = d;
    }
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return min_; }

  // Returns the current nearest.
  inline std::pair<Index, Scalar> nearest() const {
    return std::make_pair(idx_, min_);
  }

 private:
  Index idx_;
  Scalar min_;
};

template <typename Index, typename Scalar>
class SearchKnn {
 public:
  SearchKnn(Index const k, std::vector<std::pair<Index, Scalar>>* knn)
      : k_{k}, knn_{*knn}, min_{std::numeric_limits<Scalar>::max()} {
    knn_.clear();
    // Initial search distance.
    knn_.emplace_back(0, min_);
  }

  //! \private
  ~SearchKnn() {
    // If we couldn't find k neighbors, we remove the max one.
    if (knn_.back().second == std::numeric_limits<Scalar>::max()) {
      knn_.pop_back();
    }
  }

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    if (knn_.size() < k_) {
      knn_.emplace_back();
      InsertSorted(idx, d);
    } else if (d < min_) {
      InsertSorted(idx, d);
      min_ = knn_.back().second;
    }
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return min_; }

 private:
  //! \brief Inserts an element in O(n) time while keeping the vector the same
  //! length by replacing the last (furthest) element.
  //! \details It starts at the 2nd last element and moves to the front of the
  //! vector until the input distance \p d is no longer smaller than the
  //! distance being compared. While traversing the vector, each element gets
  //! copied 1 index upwards (towards end of the vector). This means we have k
  //! comparisons and k copies, where k is the amount of elements checked. This
  //! seems to dominate the Knn search time. Alternative strategies have been
  //! attempted:
  //!  * std::vector::insert(std::lower_bound).
  //!  * std::push_heap(std::vector) and std::pop_heap(std::vector).
  inline void InsertSorted(Index const idx, Scalar const d) {
    Index i;
    for (i = knn_.size() - 1; i > 0; --i) {
      if (knn_[i - 1].second > d) {
        knn_[i] = knn_[i - 1];
      } else {
        break;
      }
    }
    // We update the inserted element outside of the loop. This is done for the
    // case where we didn't break, simply reaching the end of the loop. For the
    // last element (first in the list) we can't enter the "else" clause.
    knn_[i] = std::make_pair(idx, d);
  }

  Index const k_;
  std::vector<std::pair<Index, Scalar>>& knn_;
  Scalar min_;
};

template <typename Index, typename Scalar>
class SearchRadius {
 public:
  SearchRadius(Scalar const radius, std::vector<std::pair<Index, Scalar>>* n)
      : radius_{radius}, n_{*n} {
    n_.clear();
  }

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    if (d < radius_) {
      n_.emplace_back(idx, d);
    }
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return radius_; }

 private:
  Scalar const radius_;
  std::vector<std::pair<Index, Scalar>>& n_;
};

}  // namespace internal

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

  //! Returns the nearest neighbor of point \p p .
  //! \tparam P point type.
  template <typename P>
  inline std::pair<Index, Scalar> SearchNn(P const& p) const {
    internal::SearchNn<Index, Scalar> v;
    SearchNn(p, root_, &v);
    return v.nearest();
  }

  //! Returns the \p k nearest neighbors of point \p p .
  //! \tparam P point type.
  template <typename P>
  inline void SearchKnn(
      P const& p,
      Index const k,
      std::vector<std::pair<Index, Scalar>>* knn) const {
    internal::SearchKnn<Index, Scalar> v(k, knn);
    SearchNn(p, root_, &v);
  }

  //! Returns all neighbors to point \p p that are within squared
  //! radius \p radius .
  //! \tparam P point type.
  template <typename P>
  inline void SearchRadius(
      P const& p,
      Scalar const radius,
      std::vector<std::pair<Index, Scalar>>* n) const {
    internal::SearchRadius<Index, Scalar> v(radius, n);
    SearchNn(p, root_, &v);
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

  //! Returns the nearest neighbor or neighbors of point \p p depending
  //! selection by visitor \p visitor for node \p node.
  template <typename P, typename V>
  inline void SearchNn(P const& p, Node const* const node, V* visitor) const {
    if (node->IsBranch()) {
      Scalar const v = points_(p, node->data.branch.dim);
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      if (v <= node->data.branch.split) {
        SearchNn(p, node->left, visitor);
        if (visitor->max() > std::pow(v - node->data.branch.split, 2)) {
          SearchNn(p, node->right, visitor);
        }
      } else {
        SearchNn(p, node->right, visitor);
        if (visitor->max() > std::pow(node->data.branch.split - v, 2)) {
          SearchNn(p, node->left, visitor);
        }
      }
    } else {
      (*visitor)(node->data.leaf.idx, metric_(p, node->data.leaf.idx));
    }
  }

  Points const& points_;
  Metric const metric_;
  Index const dimensions_;
  internal::ItemBuffer<Node> nodes_;
  Node* root_;
};

}  // namespace nano_tree