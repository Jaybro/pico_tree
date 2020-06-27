#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "core.hpp"

namespace nano_tree {

namespace internal {

//! KdTree search visitor for finding a single nearest neighbor.
template <typename Index, typename Scalar>
class SearchNn {
 public:
  SearchNn() : min_{std::numeric_limits<Scalar>::max()} {}

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    idx_ = idx;
    min_ = d;
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return min_; }

  //! Returns the current nearest neighbor.
  inline std::pair<Index, Scalar> nearest() const {
    return std::make_pair(idx_, min_);
  }

 private:
  Index idx_;
  Scalar min_;
};

//! \brief KdTree search visitor for finding k nearest neighbors using a max
//! heap.
template <typename Index, typename Scalar>
class SearchKnn {
 public:
  SearchKnn(Index const k, std::vector<std::pair<Index, Scalar>>* knn)
      : k_{k}, knn_{*knn} {
    // Initial search distances for the heap. All values will be replaced unless
    // point coordinates somehow have extreme values. In this case bad things
    // will happen anyway.
    knn_.assign(k_, {0, std::numeric_limits<Scalar>::max()});
  }

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) const {
    // Replace the current maximum.
    knn_[0] = std::make_pair(idx, d);
    // Repair the heap property.
    ReplaceFrontHeap(
        knn_.begin(),
        knn_.end(),
        [](std::pair<Index, Scalar> const& a, std::pair<Index, Scalar> const& b)
            -> bool { return a.second < b.second; });
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return knn_[0].second; }

 private:
  Index const k_;
  std::vector<std::pair<Index, Scalar>>& knn_;
};

//! KdTree search visitor for finding all neighbors within a radius.
template <typename Index, typename Scalar>
class SearchRadius {
 public:
  SearchRadius(Scalar const radius, std::vector<std::pair<Index, Scalar>>* n)
      : radius_{radius}, n_{*n} {
    n_.clear();
  }

  //! Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    n_.emplace_back(idx, d);
  }

  //! Maximum search distance with respect to the query point.
  inline Scalar max() const { return radius_; }

 private:
  Scalar const radius_;
  std::vector<std::pair<Index, Scalar>>& n_;
};

}  // namespace internal

//! \brief L2 metric using the squared L2 norm for measuring distances between
//! points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
template <typename Index, typename Scalar, int Dims, typename Points>
class MetricL2 {
 public:
  MetricL2(Points const& points) : points_{points} {}

  //! Calculates the difference between two points given a query point and an
  //! index to a point.
  //! \tparam P Point type.
  //! \param p Point.
  //! \param idx Index.
  template <typename P>
  inline Scalar operator()(P const& p, Index const idx) const {
    Scalar d{};

    for (Index i = 0;
         i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
         ++i) {
      Scalar const v = points_(p, i) - points_(idx, i);
      d += v * v;
    }

    return d;
  }

  //! Calculates the difference between two points for a single dimension.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! Returns the squared distance of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }

 private:
  Points const& points_;
};

//! \brief Splits a tree node using the median. Recursing down the tree results
//! in a median of medians algorithm. This strategy builds a tree in O(n log n)
//! time on average.
template <typename Index, typename Scalar, int Dims, typename Points>
class SplitterMedian {
 public:
  SplitterMedian(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  inline void operator()(
      Index const depth,
      Index const offset,
      Index const size,
      Index* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Points const& points = points_;
    *split_dim =
        depth % internal::Dimensions<Dims>::Dims(points_.num_dimensions());
    *split_idx = size / 2 + offset;

    std::nth_element(
        indices_.begin() + offset,
        indices_.begin() + *split_idx,
        indices_.begin() + offset + size,
        [&points, dim = *split_dim](Index const a, Index const b) -> bool {
          return points(a, dim) < points(b, dim);
        });

    *split_val = points(indices_[*split_idx], *split_dim);
  }

 private:
  Points const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dims The amount of spatial dimensions of the tree and points.
//! nano_tree::Dynamic in case of run time dimensions.
template <
    typename Index,
    typename Scalar,
    int Dims,
    typename Points,
    typename Metric = MetricL2<Index, Scalar, Dims, Points>,
    typename Splitter = SplitterMedian<Index, Scalar, Dims, Points>>
class KdTree {
 private:
  //! KdTree Node.
  struct Node {
    //! Data is used to either store branch or leaf information. Which union
    //! member is used can be tested with IsBranch() or IsLeaf().
    union Data {
      //! Tree branch.
      struct Branch {
        Index split_dim;
        Scalar split_val;
      };

      //! Tree leaf.
      struct Leaf {
        Index begin_idx;
        Index end_idx;
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
  //! Creates a KdTree given \p points and \p max_leaf_size. Each duplication of
  //! \p max_leaf_size reduces the height of the tree by one. This means that
  //! increasing \p max_leaf_size from, for example, 8 to 14 has little effect
  //! on the contents of the leaves (it may reducing the tree size by a single
  //! node).
  KdTree(Points const& points, Index const max_leaf_size)
      : points_{points},
        metric_{points_},
        nodes_{
            internal::MaxNodesFromPoints(points_.num_points(), max_leaf_size)},
        indices_(points_.num_points()),
        root_{MakeTree(max_leaf_size)} {
    assert(points_.num_points() > 0);
    assert(max_leaf_size > 0);
  }

  //! Returns the nearest neighbor of point \p p in O(log n) average time.
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
    // If it happens that the point set is has less points than k we just return
    // all points in the set.
    internal::SearchKnn<Index, Scalar> v(
        std::min(k, points_.num_points()), knn);
    SearchNn(p, root_, &v);
  }

  //! Returns all neighbors to point \p p that are within squared radius \p
  //! radius.
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
  //! \brief Builds a tree given a \p max_leaf_size and a Splitter.
  //! \details Run time may vary depending on the split strategy.
  inline Node* MakeTree(Index const max_leaf_size) {
    std::iota(indices_.begin(), indices_.end(), 0);
    Splitter v(points_, &indices_);
    return SplitIndices(max_leaf_size, 0, 0, points_.num_points(), v);
  }

  //! Creates a tree node for a range of indices, splits the range in two and
  //! recursively does the same for each sub set of indices until the index
  //! range \p size is less than or equal to \p max_leaf_size .
  inline Node* SplitIndices(
      Index const max_leaf_size,
      Index const depth,
      Index const offset,
      Index const size,
      Splitter const& visitor) {
    Node* node = nodes_.MakeItem();
    //
    if (size <= max_leaf_size) {
      node->data.leaf.begin_idx = offset;
      node->data.leaf.end_idx = offset + size;
      node->left = nullptr;
      node->right = nullptr;
    } else {
      Index split_idx;
      visitor(
          depth,
          offset,
          size,
          &node->data.branch.split_dim,
          &split_idx,
          &node->data.branch.split_val);
      // The split_idx is used as the first index of the right branch.
      Index const left_size = split_idx - offset;
      Index const right_size = size - left_size;

      node->left =
          SplitIndices(max_leaf_size, depth + 1, offset, left_size, visitor);
      node->right = SplitIndices(
          max_leaf_size, depth + 1, split_idx, right_size, visitor);
    }

    return node;
  }

  //! Returns the nearest neighbor or neighbors of point \p p depending
  //! selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNn(P const& p, Node const* const node, V* visitor) const {
    if (node->IsBranch()) {
      Scalar const v = points_(p, node->data.branch.split_dim);
      Scalar const d = metric_(node->data.branch.split_val, v);
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      if (v <= node->data.branch.split_val) {
        SearchNn(p, node->left, visitor);
        if (visitor->max() >= d) {
          SearchNn(p, node->right, visitor);
        }
      } else {
        SearchNn(p, node->right, visitor);
        if (visitor->max() >= d) {
          SearchNn(p, node->left, visitor);
        }
      }
    } else {
      // TODO The radius search has a stable max(). Perhaps template this point
      // visitation.
      Scalar max = visitor->max();
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Index const idx = indices_[i];
        Scalar const d = metric_(p, idx);
        if (max > d) {
          (*visitor)(idx, d);
          max = visitor->max();
        }
      }
    }
  }

  Points const& points_;
  Metric const metric_;
  internal::ItemBuffer<Node> nodes_;
  std::vector<Index> indices_;
  Node* root_;
};

}  // namespace nano_tree
