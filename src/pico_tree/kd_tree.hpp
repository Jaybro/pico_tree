#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "core.hpp"

namespace pico_tree {

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
        // TODO Perhaps store local object?
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
  inline void operator()(Index const idx, Scalar const d) const {
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

//! \brief Splits a tree node on the median of the active dimension.
//! \details The dimension is cycled while recursing down the tree. These steps
//! result in a median of medians algorithm. The tree is build in O(n log n)
//! time on average.
template <typename Index, typename Scalar, int Dims, typename Points>
class SplitterMedian {
 private:
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dims>;

 public:
  SplitterMedian(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  inline void operator()(
      Index const depth,
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
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

//! \brief Splits a tree node halfway the "fattest" axis of the bounding box
//! that contains it.
//! \details Bases on the paper "It's okay to be skinny, if your friends are
//! fat". The aspect ratio of the split is at most 2:1 unless that results in an
//! empty leaf. Then at least one point is moved into the empty leaf and the
//! split is adjusted.
//!
//! * http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf
//!
//! The tree is build in O(n log n) time and tree creation is in practice faster
//! than using SplitterMedian.
template <typename Index, typename Scalar, int Dims, typename Points>
class SplitterSlidingMidpoint {
 private:
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dims>;

 public:
  SplitterSlidingMidpoint(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  inline void operator()(
      Index const depth,
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
      Index* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    // TODO Is starting from 1 useful?
    // See which dimension of the box is the "fattest".
    Scalar max_delta = std::numeric_limits<Scalar>::lowest();
    for (Index i = 0;
         i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
         ++i) {
      Scalar const delta = box_max[i] - box_min[i];
      if (delta > max_delta) {
        max_delta = delta;
        *split_dim = i;
      }
    }
    *split_val = max_delta / Scalar(2.0) + box_min[*split_dim];

    // Everything smaller than split_val goes left, the rest right.
    Points const& points = points_;
    auto comp = [&points, dim = *split_dim, val = *split_val](
                    Index const a) -> bool { return points(a, dim) < val; };
    std::partition(
        indices_.begin() + offset, indices_.begin() + offset + size, comp);
    *split_idx = std::partition_point(
                     indices_.cbegin() + offset,
                     indices_.cbegin() + offset + size,
                     comp) -
                 indices_.cbegin();

    // If it happens that either all points are on the left side or right side,
    // one point slides to the other side and we split on the first right value
    // instead of the middle split.
    if ((*split_idx - offset) == size) {
      (*split_idx)--;
      (*split_val) = points(indices_[*split_idx], *split_dim);
    } else if ((*split_idx - offset) == 0) {
      (*split_idx)++;
      (*split_val) = points(indices_[*split_idx], *split_dim);
    }
  }

 private:
  Points const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dims The amount of spatial dimensions of the tree and points.
//! pico_tree::Dynamic in case of run time dimensions.
template <
    typename Index,
    typename Scalar,
    int Dims,
    typename Points,
    typename Metric = MetricL2<Index, Scalar, Dims, Points>,
    typename Splitter = SplitterSlidingMidpoint<Index, Scalar, Dims, Points>>
class KdTree {
 private:
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dims>;

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
    Sequence box_min;
    Sequence box_max;
    Node* left;
    Node* right;
  };

  //! KdTree builder.
  class Builder {
   public:
    Builder(
        Index const max_leaf_size,
        Splitter const& splitter,
        internal::DynamicBuffer<Node>* nodes)
        : max_leaf_size_{max_leaf_size}, splitter_{splitter}, nodes_{*nodes} {}

    //! Creates a tree node for a range of indices, splits the range in two and
    //! recursively does the same for each sub set of indices until the index
    //! range \p size is less than or equal to \p max_leaf_size .
    inline Node* SplitIndices(
        Index const depth,
        Index const offset,
        Index const size,
        typename Sequence::MoveReturnType box_min,
        typename Sequence::MoveReturnType box_max) const {
      Node* node = nodes_.MakeItem();
      //
      if (size <= max_leaf_size_) {
        node->data.leaf.begin_idx = offset;
        node->data.leaf.end_idx = offset + size;
        node->box_min = box_min.Move();
        node->box_max = box_max.Move();
        node->left = nullptr;
        node->right = nullptr;
      } else {
        Index split_idx;
        splitter_(
            depth,
            offset,
            size,
            box_min,
            box_max,
            &node->data.branch.split_dim,
            &split_idx,
            &node->data.branch.split_val);
        // The split_idx is used as the first index of the right branch.
        Index const left_size = split_idx - offset;
        Index const right_size = size - left_size;

        Sequence left_box_max = box_max;
        left_box_max[node->data.branch.split_dim] = node->data.branch.split_val;

        Sequence right_box_min = box_min;
        right_box_min[node->data.branch.split_dim] =
            node->data.branch.split_val;

        node->left = SplitIndices(
            depth + 1, offset, left_size, box_min.Move(), left_box_max.Move());
        node->right = SplitIndices(
            depth + 1,
            split_idx,
            right_size,
            right_box_min.Move(),
            box_max.Move());
      }

      return node;
    }

   private:
    Index const max_leaf_size_;
    Splitter const& splitter_;
    internal::DynamicBuffer<Node>& nodes_;
  };

 public:
  //! \brief Creates a KdTree given \p points and \p max_leaf_size.
  //! \details Each duplication of \p max_leaf_size reduces the height of the
  //! tree by one. The effect it has for anything in-between depends on the tree
  //! splitting mechanism.
  KdTree(Points const& points, Index const max_leaf_size)
      : points_{points},
        metric_{points_},
        nodes_{},
        indices_(points_.num_points()),
        root_{MakeTree(max_leaf_size)} {}

  //! \brief Returns the nearest neighbor of point \p p in O(log n) average
  //! time for randomly distributed points.
  //! \tparam P point type.
  template <typename P>
  inline std::pair<Index, Scalar> SearchNn(P const& p) const {
    internal::SearchNn<Index, Scalar> v;
    SearchNn(root_, p, &v);
    return v.nearest();
  }

  //! \brief Returns the \p k nearest neighbors of point \p p .
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
    SearchNn(root_, p, &v);
  }

  //! \brief Returns all neighbors to point \p p that are within squared radius
  //! \p radius.
  //! \tparam P point type.
  template <typename P>
  inline void SearchRadius(
      P const& p,
      Scalar const radius,
      std::vector<std::pair<Index, Scalar>>* n) const {
    internal::SearchRadius<Index, Scalar> v(radius, n);
    SearchNn(root_, p, &v);
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dims)+k).
  template <typename P>
  inline void SearchRange(
      P const& min, P const& max, std::vector<Index>* i) const {
    i->clear();
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there was overlap
    // then the search is slower. So unless many queries don't intersect there
    // is no point in adding it.
    SearchRange(root_, min, max, i);
  }

  //! \brief Point set used by the tree.
  inline Points const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

 private:
  //! \brief Builds a tree given a \p max_leaf_size and a Splitter.
  //! \details Run time may vary depending on the split strategy.
  inline Node* MakeTree(Index const max_leaf_size) {
    assert(points_.num_points() > 0);
    assert(max_leaf_size > 0);

    std::iota(indices_.begin(), indices_.end(), 0);

    Sequence box_min, box_max;
    box_min.Fill(points_.num_dimensions(), std::numeric_limits<Scalar>::max());
    box_max.Fill(
        points_.num_dimensions(), std::numeric_limits<Scalar>::lowest());

    for (Index j = 0; j < points_.num_points(); ++j) {
      for (Index i = 0;
           i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
           ++i) {
        Scalar const v = points_(j, i);
        if (v < box_min[i]) {
          box_min[i] = v;
        }
        if (v > box_max[i]) {
          box_max[i] = v;
        }
      }
    }

    Splitter splitter(points_, &indices_);
    return Builder{max_leaf_size, splitter, &nodes_}.SplitIndices(
        0, 0, points_.num_points(), box_min.Move(), box_max.Move());
  }

  //! Returns the nearest neighbor (or neighbors) of point \p p depending on
  //! their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNn(Node const* const node, P const& p, V* visitor) const {
    if (node->IsBranch()) {
      Scalar const v = points_(p, node->data.branch.split_dim);
      Scalar const d = metric_(node->data.branch.split_val, v);
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      if (v <= node->data.branch.split_val) {
        SearchNn(node->left, p, visitor);
        if (visitor->max() >= d) {
          SearchNn(node->right, p, visitor);
        }
      } else {
        SearchNn(node->right, p, visitor);
        if (visitor->max() >= d) {
          SearchNn(node->left, p, visitor);
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

  //! Checks if \p p is contained in the box defined by \p min and \p max. A
  //! point on the edge considered inside the box.
  template <typename P>
  inline bool PointInBox(Sequence const& p, P const& min, P const& max) const {
    for (Index i = 0;
         i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
         ++i) {
      if (points_(min, i) > p[i] || points_(max, i) < p[i]) {
        return false;
      }
    }
    return true;
  }

  //! Checks if the point refered to by \p idx is contained in the box defined
  //! by \p min and \p max. A point on the edge considered inside the box.
  template <typename P>
  inline bool PointInBox(Index const idx, P const& min, P const& max) const {
    for (Index i = 0;
         i < internal::Dimensions<Dims>::Dims(points_.num_dimensions());
         ++i) {
      if (points_(min, i) > points_(idx, i) ||
          points_(max, i) < points_(idx, i)) {
        return false;
      }
    }
    return true;
  }

  //! Reports all indices contained by \p node.
  inline void ReportRange(
      Node const* const node, std::vector<Index>* idxs) const {
    // TODO It is probably faster if the begin_idx and end_idx are available in
    // all nodes at the price of memory by removing the union.
    if (node->IsLeaf()) {
      std::copy(
          indices_.cbegin() + node->data.leaf.begin_idx,
          indices_.cbegin() + node->data.leaf.end_idx,
          std::back_inserter(*idxs));
    } else {
      ReportRange(node->left, idxs);
      ReportRange(node->right, idxs);
    }
  }

  //! Returns all points within the box defined by \p min and \p max for \p
  //! node.
  template <typename P>
  inline void SearchRange(
      Node const* const node,
      P const& min,
      P const& max,
      std::vector<Index>* idxs) const {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Index const idx = indices_[i];
        if (PointInBox(idx, min, max)) {
          idxs->push_back(idx);
        }
      }
    } else {
      if (PointInBox(node->left->box_min, min, max) &&
          PointInBox(node->left->box_max, min, max)) {
        ReportRange(node->left, idxs);
      } else if (
          points_(min, node->data.branch.split_dim) <
          node->data.branch.split_val) {
        SearchRange(node->left, min, max, idxs);
      }

      if (PointInBox(node->right->box_min, min, max) &&
          PointInBox(node->right->box_max, min, max)) {
        ReportRange(node->right, idxs);
      } else if (
          points_(max, node->data.branch.split_dim) >
          node->data.branch.split_val) {
        SearchRange(node->right, min, max, idxs);
      }
    }
  }

  //! Point set adapter used for querying point data.
  Points const& points_;
  //! Metric used for comparing distances.
  Metric const metric_;
  //! Memory buffer for tree nodes.
  internal::DynamicBuffer<Node> nodes_;
  //! Sorted indices that refer to points inside points_.
  std::vector<Index> indices_;
  //! Root of the KdTree.
  Node* root_;
};

}  // namespace pico_tree
