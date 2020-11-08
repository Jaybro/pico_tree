#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! See which axis of the box is the longest.
template <typename Index, typename Scalar, int Dim>
inline void LongestAxisBox(
    Sequence<Scalar, Dim> const& box_min,
    Sequence<Scalar, Dim> const& box_max,
    Index* p_max_index,
    Scalar* p_max_value) {
  assert(box_min.size() == box_max.size());

  *p_max_value = std::numeric_limits<Scalar>::lowest();

  for (int i = 0; i < Dimension<Dim>::Dim(static_cast<int>(box_min.size()));
       ++i) {
    Scalar const delta = box_max[i] - box_min[i];
    if (delta > *p_max_value) {
      *p_max_index = static_cast<Index>(i);
      *p_max_value = delta;
    }
  }
}

//! \brief Compares neighbors by distance.
template <typename Index, typename Scalar>
struct NeighborComparator {
  inline bool operator()(
      std::pair<Index, Scalar> const& a,
      std::pair<Index, Scalar> const& b) const {
    return a.second < b.second;
  }
};

//! \brief KdTree search visitor for finding a single nearest neighbor.
template <typename Index, typename Scalar>
class SearchNn {
 public:
  //! \private
  inline SearchNn(std::pair<Index, Scalar>* nn) : nn_{*nn} {
    nn_.second = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const d) const {
    nn_ = std::make_pair(idx, d);
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return nn_.second; }

 private:
  std::pair<Index, Scalar>& nn_;
};

//! \brief KdTree search visitor for finding k nearest neighbors using a max
//! heap.
template <typename Index, typename Scalar>
class SearchKnn {
 public:
  //! \private
  inline SearchKnn(Index const k, std::vector<std::pair<Index, Scalar>>* knn)
      : knn_{*knn} {
    // Initial search distances for the heap. All values will be replaced unless
    // point coordinates somehow have extreme values. In this case bad things
    // will happen anyway.
    knn_.assign(k, {0, std::numeric_limits<Scalar>::max()});
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const d) const {
    // Replace the current maximum.
    knn_[0] = std::make_pair(idx, d);
    // Repair the heap property.
    ReplaceFrontHeap(
        knn_.begin(), knn_.end(), NeighborComparator<Index, Scalar>());
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void Sort() const {
    std::sort_heap(
        knn_.begin(), knn_.end(), NeighborComparator<Index, Scalar>());
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return knn_[0].second; }

 private:
  std::vector<std::pair<Index, Scalar>>& knn_;
};

//! \brief KdTree search visitor for finding all neighbors within a radius.
template <typename Index, typename Scalar>
class SearchRadius {
 public:
  //! \private
  inline SearchRadius(
      Scalar const radius, std::vector<std::pair<Index, Scalar>>* n)
      : radius_{radius}, n_{*n} {
    n_.clear();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const d) const {
    n_.push_back(std::make_pair(idx, d));
  }

  //! \brief Sort the neighbors by distance from the query point. Can be used
  //! after the search has ended.
  inline void Sort() const {
    std::sort(n_.begin(), n_.end(), NeighborComparator<Index, Scalar>());
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return radius_; }

 private:
  Scalar const radius_;
  std::vector<std::pair<Index, Scalar>>& n_;
};

}  // namespace internal

//! \brief L1 metric using the L1 norm for measuring distances between points.
//! \see MetricL2
template <typename Index, typename Scalar, int Dim, typename Points>
class MetricL1 {
 public:
  inline explicit MetricL1(Points const& points) : points_{points} {}

  //! \brief Calculates the difference between two points given a query point
  //! and an index to a point.
  //! \tparam P Point type.
  //! \param p Point.
  //! \param idx Index.
  template <typename P>
  inline typename std::enable_if<!std::is_fundamental<P>::value, Scalar>::type
  operator()(P const& p, Index const idx) const {
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(points_.num_dimensions());
         ++i) {
      d += std::abs(points_(p, i) - points_(idx, i));
    }

    return d;
  }

  //! \brief Calculates the difference between two points for a single
  //! dimension.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute distance of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }

 private:
  Points const& points_;
};

//! \brief L2 metric using the squared L2 norm for measuring distances between
//! points.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
template <typename Index, typename Scalar, int Dim, typename Points>
class MetricL2 {
 public:
  inline explicit MetricL2(Points const& points) : points_{points} {}

  //! \brief Calculates the difference between two points given a query point
  //! and an index to a point.
  //! \tparam P Point type.
  //! \param p Point.
  //! \param idx Index.
  template <typename P>
  // The enable_if forces implicit casts which are handled by
  // operator()(Scalar, Scalar).
  inline typename std::enable_if<!std::is_fundamental<P>::value, Scalar>::type
  operator()(P const& p, Index const idx) const {
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(points_.num_dimensions());
         ++i) {
      Scalar const v = points_(p, i) - points_(idx, i);
      d += v * v;
    }

    return d;
  }

  //! \brief Calculates the difference between two points for a single
  //! dimension.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared distance of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }

 private:
  Points const& points_;
};

//! \brief Splits a tree node on the median of the longest dimension.
//! \details A version of the median of medians algorithm. The tree is build in
//! O(n log n) time on average.
//!
//! Although it builds the tree slower compared to using
//! SplitterSlidingMidpoint, it will query a single nearest neighbor faster.
//! Faster queries can offset the extra build costs in scenarios such as ICP.
//!
//! Note that this splitter is not recommended when searching for more than a
//! single neighbor.
template <typename Index, typename Scalar, int Dim, typename Points>
class SplitterLongestMedian {
 private:
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dim>;

 public:
  template <typename T>
  using MemoryBuffer = internal::StaticBuffer<T>;

  SplitterLongestMedian(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
      Index* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Points const& points = points_;

    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);

    *split_idx = size / 2 + offset;

    std::nth_element(
        indices_.begin() + offset,
        indices_.begin() + *split_idx,
        indices_.begin() + offset + size,
        [&points, &split_dim](Index const a, Index const b) -> bool {
          return points(a, *split_dim) < points(b, *split_dim);
        });

    *split_val = points(indices_[*split_idx], *split_dim);
  }

 private:
  Points const& points_;
  std::vector<Index>& indices_;
};

//! \brief Splits a tree node halfway the longest axis of the bounding box that
//! contains it.
//! \details Based on the paper "It's okay to be skinny, if your friends are
//! fat". The aspect ratio of the split is at most 2:1 unless that results in an
//! empty leaf. Then at least one point is moved into the empty leaf and the
//! split is adjusted.
//!
//! * http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf
//!
//! The tree is build in O(n log n) time and tree creation is faster than using
//! SplitterLongestMedian.
//!
//! This splitter can be used to answer an approximate nearest neighbor query in
//! O(1/e^d log n) time.
template <typename Index, typename Scalar, int Dim, typename Points>
class SplitterSlidingMidpoint {
 private:
  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dim>;

 public:
  template <typename T>
  using MemoryBuffer = internal::DynamicBuffer<T>;

  SplitterSlidingMidpoint(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
      Index* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);
    *split_val = max_delta / Scalar(2.0) + box_min[*split_dim];

    // Everything smaller than split_val goes left, the rest right.
    Points const& points = points_;
    auto const comp = [&points, &split_dim, &split_val](Index const a) -> bool {
      return points(a, *split_dim) < *split_val;
    };
    std::partition(
        indices_.begin() + offset, indices_.begin() + offset + size, comp);
    *split_idx = static_cast<Index>(
        std::partition_point(
            indices_.cbegin() + offset,
            indices_.cbegin() + offset + size,
            comp) -
        indices_.cbegin());

    // If it happens that either all points are on the left side or right side,
    // one point slides to the other side and we split on the first right value
    // instead of the middle split.
    // In these two cases the split value is unknown and a partial sort is
    // required to obtain it, but also to rearrange all other indices such that
    // they are on their corresponding left or right side.
    if ((*split_idx - offset) == size) {
      (*split_idx)--;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [&points, &split_dim](Index const a, Index const b) -> bool {
            return points(a, *split_dim) < points(b, *split_dim);
          });
      (*split_val) = points(indices_[*split_idx], *split_dim);
    } else if ((*split_idx - offset) == 0) {
      (*split_idx)++;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [&points, &split_dim](Index const a, Index const b) -> bool {
            return points(a, *split_dim) < points(b, *split_dim);
          });
      (*split_val) = points(indices_[*split_idx], *split_dim);
    }
  }

 private:
  Points const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dim The spatial dimension of the tree and points. It can be set to
//! pico_tree::kDynamicDim in case Dim is only known at run-time.
template <
    typename Index,
    typename Scalar,
    int Dim,
    typename Points,
    typename Metric = MetricL2<Index, Scalar, Dim, Points>,
    typename Splitter = SplitterSlidingMidpoint<Index, Scalar, Dim, Points>>
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

    Node* left;
    Node* right;
    Data data;
  };

  //! Either an array or vector (compile time vs. run time).
  using Sequence = typename internal::Sequence<Scalar, Dim>;
  using MemoryBuffer = typename Splitter::template MemoryBuffer<Node>;

  //! KdTree builder.
  class Builder {
   public:
    Builder(
        Index const max_leaf_size,
        Splitter const& splitter,
        MemoryBuffer* nodes)
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
    MemoryBuffer& nodes_;
  };

 public:
  //! \brief Creates a KdTree given \p points and \p max_leaf_size.
  //! \details Each duplication of \p max_leaf_size reduces the height of the
  //! tree by one. The effect it has for anything in-between depends on the tree
  //! splitting mechanism.
  KdTree(Points const& points, Index const max_leaf_size)
      : points_{points},
        metric_{points_},
        nodes_(internal::MaxNodesFromPoints(points_.num_points())),
        indices_(points_.num_points()),
        root_{Build(max_leaf_size)} {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p p depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  template <typename P, typename V>
  inline void SearchNn(P const& p, V* visitor) const {
    SearchNn(root_, p, visitor);
  }

  //! \brief Searches for the nearest neighbor of point \p p .
  template <typename P>
  inline void SearchNn(P const& p, std::pair<Index, Scalar>* nn) const {
    internal::SearchNn<Index, Scalar> v(nn);
    SearchNn(root_, p, &v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p p . The output
  //! vector \p knn contains an index and distance pair for each of the search
  //! results. Interpretation of the distances depends on the Metric.
  //! \tparam P point type.
  template <typename P>
  inline void SearchKnn(
      P const& p,
      Index const k,
      std::vector<std::pair<Index, Scalar>>* knn,
      bool const sort = false) const {
    // If it happens that the point set is has less points than k we just return
    // all points in the set.
    internal::SearchKnn<Index, Scalar> v(
        std::min(k, points_.num_points()), knn);
    SearchNn(root_, p, &v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Searches for all the neighbors of point \p p that are within radius
  //! \p radius. The output vector \p n contains an index and distance pair for
  //! each of the search results.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default MetricL2 results in squared distances.
  //! \param p Input point.
  //! \param radius Search radius that depends on the Metric used by the KdTree.
  //! It should be the squared distance when using MetricL2. \code{.cpp} Scalar
  //! metric_distance = kdtree.metric()(distance); \endcode
  //! \param n Output points.
  //! \tparam P point type.
  template <typename P>
  inline void SearchRadius(
      P const& p,
      Scalar const radius,
      std::vector<std::pair<Index, Scalar>>* n,
      bool const sort = false) const {
    internal::SearchRadius<Index, Scalar> v(radius, n);
    SearchNn(root_, p, &v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dim)+k).
  template <typename P>
  inline void SearchBox(
      P const& min, P const& max, std::vector<Index>* i) const {
    i->clear();
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there was overlap
    // then the search is slower. So unless many queries don't intersect there
    // is no point in adding it.
    SearchBox(
        root_, min, max, Sequence(root_box_min_), Sequence(root_box_max_), i);
  }

  //! \brief Point set used by the tree.
  inline Points const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static KdTree Load(Points const& points, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::in | std::ios::binary);
    return Load(points, &stream);
  }

  //! \brief Loads the tree in binary from \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Does not check if the stored tree structure is valid for the given
  //! point set. \li Does not check if the stored tree structure is valid for
  //! the given template arguments.
  static KdTree Load(Points const& points, std::iostream* stream) {
    internal::Stream s(stream);
    return KdTree(points, &s);
  }

  //! \brief Saves the tree in binary to file.
  static void Save(KdTree const& tree, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::out | std::ios::binary);
    Save(tree, &stream);
  }

  //! \brief Saves the tree in binary to \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Stores the tree structure but not the points.
  static void Save(KdTree const& tree, std::iostream* stream) {
    internal::Stream s(stream);
    tree.Save(&s);
  }

 private:
  //! \brief Constructs a KdTree by reading its indexing and leaf information
  //! from a Stream.
  KdTree(Points const& points, internal::Stream* stream)
      : points_{points},
        metric_{points_},
        nodes_(internal::MaxNodesFromPoints(points_.num_points())),
        indices_(points_.num_points()),
        root_{Load(stream)} {}

  inline void CalculateBoundingBox(Sequence* p_min, Sequence* p_max) {
    Sequence& min = *p_min;
    Sequence& max = *p_max;
    min.Fill(points_.num_dimensions(), std::numeric_limits<Scalar>::max());
    max.Fill(points_.num_dimensions(), std::numeric_limits<Scalar>::lowest());

    for (Index j = 0; j < points_.num_points(); ++j) {
      for (int i = 0;
           i < internal::Dimension<Dim>::Dim(points_.num_dimensions());
           ++i) {
        Scalar const v = points_(j, i);
        if (v < min[i]) {
          min[i] = v;
        }
        if (v > max[i]) {
          max[i] = v;
        }
      }
    }
  }

  //! \brief Builds a tree given a \p max_leaf_size and a Splitter.
  //! \details Run time may vary depending on the split strategy.
  inline Node* Build(Index const max_leaf_size) {
    assert(points_.num_points() > 0);
    assert(max_leaf_size > 0);

    std::iota(indices_.begin(), indices_.end(), 0);

    CalculateBoundingBox(&root_box_min_, &root_box_max_);

    Splitter splitter(points_, &indices_);
    return Builder{max_leaf_size, splitter, &nodes_}.SplitIndices(
        0,
        0,
        points_.num_points(),
        Sequence(root_box_min_),
        Sequence(root_box_max_));
  }

  //! Returns the nearest neighbor (or neighbors) of point \p p depending on
  //! their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNn(Node const* const node, P const& p, V* visitor) const {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Scalar const d = metric_(p, indices_[i]);
        if (visitor->max() > d) {
          (*visitor)(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      Scalar const v = points_(p, node->data.branch.split_dim);
      Node const* node_1st;
      Node const* node_2nd;

      // On equals we would possibly need to go left as well. However, this is
      // handled by the if statement below this one: the check that max search
      // radius still hits the split value after having traversed the first
      // branch.
      if (v < node->data.branch.split_val) {
        node_1st = node->left;
        node_2nd = node->right;
      } else {
        node_1st = node->right;
        node_2nd = node->left;
      }

      SearchNn(node_1st, p, visitor);
      if (visitor->max() >= metric_(node->data.branch.split_val, v)) {
        SearchNn(node_2nd, p, visitor);
      }
    }
  }

  //! Checks if \p p is contained in the box defined by \p min and \p max. A
  //! point on the edge considered inside the box.
  template <typename P>
  inline bool PointInBox(Sequence const& p, P const& min, P const& max) const {
    for (Index i = 0;
         i < internal::Dimension<Dim>::Dim(points_.num_dimensions());
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
         i < internal::Dimension<Dim>::Dim(points_.num_dimensions());
         ++i) {
      Scalar const v = points_(idx, i);
      if (points_(min, i) > v || points_(max, i) < v) {
        return false;
      }
    }
    return true;
  }

  //! Reports all indices contained by \p node.
  inline void ReportNode(
      Node const* const node, std::vector<Index>* idxs) const {
    if (node->IsLeaf()) {
      std::copy(
          indices_.cbegin() + node->data.leaf.begin_idx,
          indices_.cbegin() + node->data.leaf.end_idx,
          std::back_inserter(*idxs));
    } else {
      ReportNode(node->left, idxs);
      ReportNode(node->right, idxs);
    }
  }

  //! \brief Returns all points within the box defined by \p rng_min and \p
  //! rng_max for \p node. Query time is bounded by O(n^(1-1/Dim)+k).
  //! \details Many tree nodes are excluded by checking if they intersect with
  //! the box of the query. We don't store the bounding box of each node but
  //! calculate them at run time. This slows down SearchBox in favor of
  //! SearchNn.
  template <typename P>
  inline void SearchBox(
      Node const* const node,
      P const& rng_min,
      P const& rng_max,
      typename Sequence::MoveReturnType box_min,
      typename Sequence::MoveReturnType box_max,
      std::vector<Index>* idxs) const {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Index const idx = indices_[i];
        if (PointInBox(idx, rng_min, rng_max)) {
          idxs->push_back(idx);
        }
      }
    } else {
      Sequence left_box_max = box_max;
      left_box_max[node->data.branch.split_dim] = node->data.branch.split_val;

      // Check if the left node is fully contained. If true, report all its
      // indices. Else, if its partially contained, continue the range search
      // down the left node.
      if (PointInBox(box_min, rng_min, rng_max) &&
          PointInBox(left_box_max, rng_min, rng_max)) {
        ReportNode(node->left, idxs);
      } else if (
          points_(rng_min, node->data.branch.split_dim) <
          node->data.branch.split_val) {
        SearchBox(
            node->left,
            rng_min,
            rng_max,
            box_min.Move(),
            left_box_max.Move(),
            idxs);
      }

      Sequence right_box_min = box_min;
      right_box_min[node->data.branch.split_dim] = node->data.branch.split_val;

      // Same as the left side.
      if (PointInBox(right_box_min, rng_min, rng_max) &&
          PointInBox(box_max, rng_min, rng_max)) {
        ReportNode(node->right, idxs);
      } else if (
          points_(rng_max, node->data.branch.split_dim) >
          node->data.branch.split_val) {
        SearchBox(
            node->right,
            rng_min,
            rng_max,
            right_box_min.Move(),
            box_max.Move(),
            idxs);
      }
    }
  }

  //! \brief Recursively reads the Node and its descendants.
  inline Node* ReadNode(internal::Stream* stream) {
    Node* node = nodes_.MakeItem();
    bool is_leaf;
    stream->Read(&is_leaf);

    if (is_leaf) {
      stream->Read(&node->data.leaf.begin_idx);
      stream->Read(&node->data.leaf.end_idx);
      node->left = nullptr;
      node->right = nullptr;
    } else {
      stream->Read(&node->data.branch.split_dim);
      stream->Read(&node->data.branch.split_val);
      node->left = ReadNode(stream);
      node->right = ReadNode(stream);
    }

    return node;
  }

  //! \brief Recursively writes the Node and its descendants.
  inline void WriteNode(
      Node const* const node, internal::Stream* stream) const {
    if (node->IsLeaf()) {
      stream->Write(true);
      stream->Write(node->data.leaf.begin_idx);
      stream->Write(node->data.leaf.end_idx);
    } else {
      stream->Write(false);
      stream->Write(node->data.branch.split_dim);
      stream->Write(node->data.branch.split_val);
      WriteNode(node->left, stream);
      WriteNode(node->right, stream);
    }
  }

  //! \private
  inline Node* Load(internal::Stream* stream) {
    stream->Read(&indices_);
    stream->Read(&root_box_min_);
    stream->Read(&root_box_max_);
    return ReadNode(stream);
  }

  //! \private
  inline void Save(internal::Stream* stream) const {
    stream->Write(indices_);
    stream->Write(root_box_min_);
    stream->Write(root_box_max_);
    WriteNode(root_, stream);
  }

  //! Point set adapter used for querying point data.
  Points const& points_;
  //! Metric used for comparing distances.
  Metric const metric_;
  //! Memory buffer for tree nodes.
  MemoryBuffer nodes_;
  //! Sorted indices that refer to points inside points_.
  std::vector<Index> indices_;
  //! Min coordinate of the root node box.
  Sequence root_box_min_;
  //! Max coordinate of the root node box.
  Sequence root_box_max_;
  //! Root of the KdTree.
  Node const* const root_;
};

}  // namespace pico_tree
