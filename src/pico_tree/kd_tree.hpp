#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "core.hpp"

namespace pico_tree {

namespace internal {

//! \brief See which axis of the box is the longest.
template <typename Scalar, int Dim>
inline void LongestAxisBox(
    Sequence<Scalar, Dim> const& box_min,
    Sequence<Scalar, Dim> const& box_max,
    int* p_max_index,
    Scalar* p_max_value) {
  assert(box_min.size() == box_max.size());

  *p_max_value = std::numeric_limits<Scalar>::lowest();

  for (int i = 0; i < Dimension<Dim>::Dim(static_cast<int>(box_min.size()));
       ++i) {
    Scalar const delta = box_max[i] - box_min[i];
    if (delta > *p_max_value) {
      *p_max_index = i;
      *p_max_value = delta;
    }
  }
}

//! \brief Compares neighbors by distance.
template <typename Index, typename Scalar>
struct NeighborComparator {
  //! \private
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

//! \brief KdTree search visitor for finding k nearest neighbors using an
//! insertion sort.
//! \details This rather brute-force method of maintaining a sorted sequence for
//! keeping track of the k nearest neighbors performs fast in practice. This is
//! likely due to points being reasonably ordered by the KdTree. The following
//! strategies have been attempted:
//!  * std::vector::insert(std::lower_bound) - the predecessor of the current
//!  version.
//!  * std::push_heap(std::vector) and std::pop_heap(std::vector).
//!  * std::push_heap(std::vector) followed by a custom ReplaceFrontHeap once
//!  the heap reached size k. This is the fastest "priority queue" version so
//!  far. Even without sorting the heap it is still slower than maintaining a
//!  sorted sequence.
template <typename RandomAccessIterator>
class SearchKnn {
 private:
  static_assert(
      std::is_same<
          typename std::iterator_traits<
              RandomAccessIterator>::iterator_category,
          std::random_access_iterator_tag>::value,
      "SEARCH_KNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using Pair = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Index = typename Pair::first_type;
  using Scalar = typename Pair::second_type;

 public:
  //! \private
  inline SearchKnn(RandomAccessIterator begin, RandomAccessIterator end)
      : begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->second = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    InsertSorted(
        begin_, active_end_, Pair{idx, d}, NeighborComparator<Index, Scalar>());
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return std::prev(end_)->second; }

 private:
  RandomAccessIterator begin_;
  RandomAccessIterator end_;
  RandomAccessIterator active_end_;
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

//! \brief Search visitor for finding approximate nearest neighbors.
//! \details Points and tree nodes are skipped by scaling down the search
//! distance, possibly not visiting the true nearest neighbor. An approximate
//! nearest neighbor will at most be a factor of distance ratio \p e farther
//! from the query point than the true nearest neighbor: max_ann_distance =
//! true_nn_distance * e.
//!
//! There are different possible implementations to get an approximate nearest
//! neighbor but this one is (probably) the cheapest by skipping both points
//! inside leafs and complete tree nodes. Even though all points are checked
//! inside a leaf, not all of them are visited. This saves on scaling and heap
//! updates.
//! \see SearchKnn
template <typename RandomAccessIterator>
class SearchAknn {
 private:
  static_assert(
      std::is_same<
          typename std::iterator_traits<
              RandomAccessIterator>::iterator_category,
          std::random_access_iterator_tag>::value,
      "SEARCH_AKNN_EXPECTED_RANDOM_ACCESS_ITERATOR");

  using Pair = typename std::iterator_traits<RandomAccessIterator>::value_type;
  using Index = typename Pair::first_type;
  using Scalar = typename Pair::second_type;

 public:
  //! \private
  inline SearchAknn(
      Scalar const e, RandomAccessIterator begin, RandomAccessIterator end)
      : re_{Scalar(1.0) / e}, begin_{begin}, end_{end}, active_end_{begin} {
    // Initial search distance that gets updated once k neighbors have been
    // found.
    std::prev(end_)->second = std::numeric_limits<Scalar>::max();
  }

  //! \brief Visit current point.
  inline void operator()(Index const idx, Scalar const d) {
    if (active_end_ < end_) {
      ++active_end_;
    }

    // Replace the current maximum for which the distance is scaled to be:
    // d = d / e.
    InsertSorted(
        begin_,
        active_end_,
        Pair{idx, d * re_},
        NeighborComparator<Index, Scalar>());
  }

  //! \brief Maximum search distance with respect to the query point.
  inline Scalar const& max() const { return std::prev(end_)->second; }

 private:
  Scalar re_;
  RandomAccessIterator begin_;
  RandomAccessIterator end_;
  RandomAccessIterator active_end_;
};

}  // namespace internal

//! \brief L1 metric using the L1 norm for measuring distances between points.
//! \see MetricL2
template <typename Scalar, int Dim>
class MetricL1 {
 public:
  //! \brief Creates a MetricL1 given a spatial dimension.
  inline explicit MetricL1(int const dim) : dim_{dim} {}

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(Scalar, Scalar).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      Scalar>::type
  operator()(P0 const& p0, P1 const& p1) const {
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(dim_); ++i) {
      d += std::abs(p0(i) - p1(i));
    }

    return d;
  }

  //! \brief Calculates the difference between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    return std::abs(x - y);
  }

  //! \brief Returns the absolute value of \p x.
  inline Scalar operator()(Scalar const x) const { return std::abs(x); }

 private:
  int const dim_;
};

//! \brief The L2 metric measures distances between points using the squared L2
//! norm.
//! \details For more details:
//! * https://en.wikipedia.org/wiki/Metric_space
//! * https://en.wikipedia.org/wiki/Lp_space
template <typename Scalar, int Dim>
class MetricL2 {
 public:
  //! \brief Creates a MetricL2 given a spatial dimension.
  inline explicit MetricL2(int const dim) : dim_{dim} {}

  //! \brief Calculates the distance between points \p p0 and \p p1.
  //! \tparam P0 Point type.
  //! \tparam P1 Point type.
  //! \param p0 Point.
  //! \param p1 Point.
  template <typename P0, typename P1>
  // The enable_if is not required but it forces implicit casts which are
  // handled by operator()(Scalar, Scalar).
  inline typename std::enable_if<
      !std::is_fundamental<P0>::value && !std::is_fundamental<P1>::value,
      Scalar>::type
  operator()(P0 const& p0, P1 const& p1) const {
    Scalar d{};

    for (int i = 0; i < internal::Dimension<Dim>::Dim(dim_); ++i) {
      Scalar const v = p0(i) - p1(i);
      d += v * v;
    }

    return d;
  }

  //! \brief Calculates the difference between two coordinates.
  inline Scalar operator()(Scalar const x, Scalar const y) const {
    Scalar const d = x - y;
    return d * d;
  }

  //! \brief Returns the squared value of \p x.
  inline Scalar operator()(Scalar const x) const { return x * x; }

 private:
  int const dim_;
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
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::StaticBuffer<T>;

  //! \private
  SplitterLongestMedian(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
      int* split_dim,
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
          return points(a)(*split_dim) < points(b)(*split_dim);
        });

    *split_val = points(indices_[*split_idx])(*split_dim);
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
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::DynamicBuffer<T>;

  //! \private
  SplitterSlidingMidpoint(Points const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence const& box_min,
      Sequence const& box_max,
      int* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);
    *split_val = max_delta / Scalar(2.0) + box_min[*split_dim];

    // Everything smaller than split_val goes left, the rest right.
    Points const& points = points_;
    auto const comp = [&points, &split_dim, &split_val](Index const a) -> bool {
      return points(a)(*split_dim) < *split_val;
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
            return points(a)(*split_dim) < points(b)(*split_dim);
          });
      (*split_val) = points(indices_[*split_idx])(*split_dim);
    } else if ((*split_idx - offset) == 0) {
      (*split_idx)++;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [&points, &split_dim](Index const a, Index const b) -> bool {
            return points(a)(*split_dim) < points(b)(*split_dim);
          });
      (*split_val) = points(indices_[*split_idx])(*split_dim);
    }
  }

 private:
  Points const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dim The spatial dimension of the tree. It can be set to
//! pico_tree::kDynamicDim in case Dim is only known at run-time.
template <
    typename Index,
    typename Scalar,
    int Dim,
    typename Points,
    typename Metric = MetricL2<Scalar, Dim>,
    typename Splitter = SplitterSlidingMidpoint<Index, Scalar, Dim, Points>>
class KdTree {
 private:
  //! \brief KdTree Node.
  struct Node {
    //! \brief Data is used to either store branch or leaf information. Which
    //! union member is used can be tested with IsBranch() or IsLeaf().
    union Data {
      //! \brief Tree branch.
      struct Branch {
        //! \brief Split coordinate / index of the KdTree spatial dimension.
        int split_dim;
        //! \brief Coordinate value used for splitting the children of a node.
        Scalar split_val;
      };

      //! \brief Tree leaf.
      struct Leaf {
        //! \private
        Index begin_idx;
        //! \private
        Index end_idx;
      };

      //! \brief Union branch data.
      Branch branch;
      //! \brief Union leaf data.
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
  //! \brief The KdTree cannot be copied.
  //! \details The KdTree uses pointers to refer to tree nodes. These would all
  //! be invalidated during a deep copy.
  //! \private
  KdTree(KdTree const&) = delete;

  //! \brief Move constructor of the KdTree.
  //! \details The move constructor is not implicitly created because of the
  //! deleted copy constructor.
  //! \private
  KdTree(KdTree&&) = default;

  //! \brief Creates a KdTree given \p points and \p max_leaf_size.
  //!
  //! \details
  //! The KdTree wants ownership of \p points to avoid problems that may occur
  //! when keeping a const reference to the data. For example, using std::move()
  //! on the point set would invalidate the local reference which is then
  //! unrecoverable.
  //!
  //! To avoid a deep copy of the \p points object:
  //! \li Move it: KdTree tree(std::move(points), max_leaf_size);
  //! \li Implement its class as an adaptor that keeps a reference to the data.
  //!
  //! The value of \p max_leaf_size influences the height and performance of the
  //! tree. The splitting mechanism determines data locality within the leafs.
  //! The exact effect it has depends on the tree splitting mechanism.
  //!
  //! \param points The input point set (interface).
  //! \param max_leaf_size The maximum amount of points allowed in a leaf node.
  KdTree(Points points, Index max_leaf_size)
      : points_(std::move(points)),
        metric_(points_.sdim()),
        nodes_(internal::MaxNodesFromPoints(points_.npts())),
        indices_(points_.npts()),
        root_(Build(max_leaf_size)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p p depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNn(P const& p, V* visitor) const {
    SearchNn(root_, p, visitor);
  }

  //! \brief Searches for the nearest neighbor of point \p p .
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default MetricL2 results in a squared distance.
  template <typename P>
  inline void SearchNn(P const& p, std::pair<Index, Scalar>* nn) const {
    internal::SearchNn<Index, Scalar> v(nn);
    SearchNn(root_, p, &v);
  }

  //! \brief Searches for the k nearest neighbors of point \p p , where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals std::pair<Index, Scalar>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default MetricL2 results in squared distances.
  //! \tparam P Point type.
  //! \tparam RandomAccessIterator Iterator type.
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& p, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            std::pair<Index, Scalar>>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_PAIR_INDEX_SCALAR");

    internal::SearchKnn<RandomAccessIterator> v(begin, end);
    SearchNn(root_, p, &v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p p . The output
  //! vector \p knn contains an index and distance pair for each of the search
  //! results.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& p,
      Index const k,
      std::vector<std::pair<Index, Scalar>>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, points_.npts()));
    SearchKnn(p, knn->begin(), knn->end());
  }

  //! \brief Searches for all the neighbors of point \p p that are within radius
  //! \p radius. The output vector \p n contains an index and distance pair for
  //! each of the search results.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default MetricL2 results in squared distances.
  //! \tparam P Point type.
  //! \param p Input point.
  //! \param radius Search radius. The interpretation of the radius depends on
  //! the Metric used by the KdTree. Squared distance are required when using
  //! MetricL2.
  //! \code{.cpp}
  //! Scalar distance = -2.0;
  //! // E.g., MetricL1: 2.0, MetricL2: 4.0
  //! Scalar metric_distance = kdtree.metric()(distance);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
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

  //! \brief Searches for the k approximate nearest neighbors of point \p p ,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals std::pair<Index, Scalar>.
  //! \details This function can result in faster search queries compared to
  //! KdTree::SearchKnn by skipping points and tree nodes. This is achieved by
  //! scaling down the search distance, possibly not visiting the true nearest
  //! neighbor. An approximate nearest neighbor will at most be a factor of
  //! distance ratio \p e farther from the query point than the true nearest
  //! neighbor: max_ann_distance = true_nn_distance * e. This holds true for
  //! each respective nn index i, 0 <= i < k.
  //!
  //! The amount of requested neighbors, k, should be sufficiently large to get
  //! a noticeable speed increase from this method. Within a leaf all points are
  //! compared to the query anyway, even if they are skipped. These calculations
  //! can be avoided by skipping leafs completely, which will never happen if
  //! all requested neighbors reside within a single one.
  //!
  //! Interpretation of both the input error ratio and output distances
  //! depend on the Metric. The default MetricL2 calculates squared
  //! distances. Using this metric, the input error ratio should be the squared
  //! error ratio and the output distances will be squared distances scaled by
  //! the inverse error ratio.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! Scalar max_error = Scalar(0.15);
  //! Scalar e = tree.metric()(Scalar(1.0) + max_error);
  //! std::vector<std::pair<Index, Scalar>> knn(k);
  //! tree.SearchAknn(p, e, knn.begin(), knn.end());
  //! // Optionally scale back to the actual metric distance.
  //! for (auto& nn : knn) { nn.second *= e; }
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchAknn(
      P const& p,
      Scalar const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            std::pair<Index, Scalar>>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_PAIR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNn(root_, p, &v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p p .
  //! The output vector \p knn contains an index and distance pair for each of
  //! the search results.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchAknn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchAknn(
      P const& p,
      Index const k,
      Scalar const e,
      std::vector<std::pair<Index, Scalar>>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, points_.npts()));
    SearchAknn(p, e, knn->begin(), knn->end());
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dim)+k).
  //! \tparam P Point type.
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
  static KdTree Load(Points points, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::in | std::ios::binary);
    return Load(std::move(points), &stream);
  }

  //! \brief Loads the tree in binary from \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Does not check if the stored tree structure is valid for the given
  //! point set. \li Does not check if the stored tree structure is valid for
  //! the given template arguments.
  static KdTree Load(Points points, std::iostream* stream) {
    internal::Stream s(stream);
    return KdTree(std::move(points), &s);
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
  KdTree(Points points, internal::Stream* stream)
      : points_(std::move(points)),
        metric_(points_.sdim()),
        nodes_(internal::MaxNodesFromPoints(points_.npts())),
        indices_(points_.npts()),
        root_(Load(stream)) {}

  inline void CalculateBoundingBox(Sequence* p_min, Sequence* p_max) {
    Sequence& min = *p_min;
    Sequence& max = *p_max;
    min.Fill(points_.sdim(), std::numeric_limits<Scalar>::max());
    max.Fill(points_.sdim(), std::numeric_limits<Scalar>::lowest());

    for (Index j = 0; j < points_.npts(); ++j) {
      auto const& p = points_(j);
      for (int i = 0; i < internal::Dimension<Dim>::Dim(points_.sdim()); ++i) {
        Scalar const v = p(i);
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
    assert(points_.npts() > 0);
    assert(max_leaf_size > 0);

    std::iota(indices_.begin(), indices_.end(), 0);

    CalculateBoundingBox(&root_box_min_, &root_box_max_);

    Splitter splitter(points_, &indices_);
    return Builder{max_leaf_size, splitter, &nodes_}.SplitIndices(
        0, 0, points_.npts(), Sequence(root_box_min_), Sequence(root_box_max_));
  }

  //! Returns the nearest neighbor (or neighbors) of point \p p depending on
  //! their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNn(Node const* const node, P const& p, V* visitor) const {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Scalar const d = metric_(p, points_(indices_[i]));
        if (visitor->max() > d) {
          (*visitor)(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      Scalar const v = p(node->data.branch.split_dim);
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
    for (int i = 0; i < internal::Dimension<Dim>::Dim(points_.sdim()); ++i) {
      if (min(i) > p[i] || max(i) < p[i]) {
        return false;
      }
    }
    return true;
  }

  //! Checks if the point refered to by \p idx is contained in the box defined
  //! by \p min and \p max. A point on the edge considered inside the box.
  template <typename P>
  inline bool PointInBox(Index const idx, P const& min, P const& max) const {
    auto const& p = points_(idx);
    for (int i = 0; i < internal::Dimension<Dim>::Dim(points_.sdim()); ++i) {
      Scalar const v = p(i);
      if (min(i) > v || max(i) < v) {
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
          rng_min(node->data.branch.split_dim) < node->data.branch.split_val) {
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
          rng_max(node->data.branch.split_dim) > node->data.branch.split_val) {
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
  Points points_;
  //! Metric used for comparing distances.
  Metric metric_;
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
