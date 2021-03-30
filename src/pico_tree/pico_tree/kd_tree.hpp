#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>

#include "pico_tree/internal/memory.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/sequence.hpp"
#include "pico_tree/internal/stream.hpp"
#include "pico_tree/metric.hpp"

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

}  // namespace internal

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
template <typename Traits>
class SplitterLongestMedian {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  static constexpr int Dim = Traits::Dim;
  using Space = typename Traits::SpaceType;
  template <int Dim_>
  using Sequence = typename internal::Sequence<Scalar, Dim_>;

 public:
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::StaticBuffer<T>;

  //! \private
  SplitterLongestMedian(Space const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  template <int Dim_>
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence<Dim_> const& box_min,
      Sequence<Dim_> const& box_max,
      int* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    auto const& points = points_;

    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);

    *split_idx = size / 2 + offset;

    std::nth_element(
        indices_.begin() + offset,
        indices_.begin() + *split_idx,
        indices_.begin() + offset + size,
        [this, &split_dim](Index const a, Index const b) -> bool {
          return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
        });

    *split_val = PointCoord(indices_[*split_idx], *split_dim);
  }

 private:
  inline Scalar const& PointCoord(
      Index const point_idx, int const coord_idx) const {
    return Traits::PointCoords(Traits::PointAt(points_, point_idx))[coord_idx];
  }

  Space const& points_;
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
template <typename Traits>
class SplitterSlidingMidpoint {
 private:
  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;
  template <int Dim_>
  using Sequence = typename internal::Sequence<Scalar, Dim_>;

 public:
  //! \brief Buffer type used with this splitter.
  template <typename T>
  using MemoryBuffer = internal::DynamicBuffer<T>;

  //! \private
  SplitterSlidingMidpoint(Space const& points, std::vector<Index>* p_indices)
      : points_{points}, indices_{*p_indices} {}

  //! \brief This function splits a node.
  template <int Dim_>
  inline void operator()(
      Index const,  // depth
      Index const offset,
      Index const size,
      Sequence<Dim_> const& box_min,
      Sequence<Dim_> const& box_max,
      int* split_dim,
      Index* split_idx,
      Scalar* split_val) const {
    Scalar max_delta;
    internal::LongestAxisBox(box_min, box_max, split_dim, &max_delta);
    *split_val = max_delta / Scalar(2.0) + box_min[*split_dim];

    // Everything smaller than split_val goes left, the rest right.
    auto const comp = [this, &split_dim, &split_val](Index const a) -> bool {
      return PointCoord(a, *split_dim) < *split_val;
    };

    *split_idx = static_cast<Index>(
        std::partition(
            indices_.begin() + offset, indices_.begin() + offset + size, comp) -
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
          [this, &split_dim](Index const a, Index const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(indices_[*split_idx], *split_dim);
    } else if ((*split_idx - offset) == 0) {
      (*split_idx)++;
      std::nth_element(
          indices_.begin() + offset,
          indices_.begin() + (*split_idx),
          indices_.begin() + offset + size,
          [this, &split_dim](Index const a, Index const b) -> bool {
            return PointCoord(a, *split_dim) < PointCoord(b, *split_dim);
          });
      (*split_val) = PointCoord(indices_[*split_idx], *split_dim);
    }
  }

 private:
  inline Scalar const& PointCoord(Index point_idx, int coord_idx) const {
    return Traits::PointCoords(Traits::PointAt(points_, point_idx))[coord_idx];
  }

  Space const& points_;
  std::vector<Index>& indices_;
};

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
//! \tparam Dim_ The spatial dimension of the tree. Dim_ defaults to Traits::Dim
//! but can be set to a different value should a rare occasion require it.
template <
    typename Traits,
    typename Metric = L2Squared<Traits>,
    typename Splitter = SplitterSlidingMidpoint<Traits>,
    int Dim_ = Traits::Dim>
class KdTree {
 private:
  static_assert(
      Dim_ <= Traits::Dim,
      "SPATIAL_DIMENSION_TREE_MUST_BE_SMALLER_OR_EQUAL_TO_TRAITS_DIMENSION");

  using Index = typename Traits::IndexType;
  using Scalar = typename Traits::ScalarType;
  using Space = typename Traits::SpaceType;

 public:
  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = Scalar;
  //! \brief KdTree dimension. It equals pico_tree::kDynamicDim in case Dim is
  //! only known at run-time.
  static constexpr int Dim = Dim_;
  //! \brief Traits with information about the input Spaces and Points.
  using TraitsType = Traits;
  //! \brief Point set or adaptor type.
  using SpaceType = Space;
  //! \brief The metric used for various searches.
  using MetricType = Metric;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<Index, Scalar>;

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
      Node* node = nodes_.Allocate();
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
  //! \details The KdTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy. For now we are not interested in
  //! providing a deep copy.
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
  KdTree(Space points, Index max_leaf_size)
      : points_(std::move(points)),
        metric_(),
        nodes_(internal::MaxNodesFromPoints(Traits::SpaceNpts(points_))),
        indices_(Traits::SpaceNpts(points_)),
        root_(Build(max_leaf_size)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNearest(P const& x, V* visitor) const {
    SearchNearest(root_, x, visitor);
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default L2Squared results in a squared distance.
  template <typename P>
  inline void SearchNn(P const& x, NeighborType* nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(root_, x, &v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<Index, Scalar>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \tparam RandomAccessIterator Iterator type.
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& x, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchKnn<RandomAccessIterator> v(begin, end);
    SearchNearest(root_, x, &v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x, Index const k, std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits::SpaceNpts(points_)));
    SearchKnn(x, knn->begin(), knn->end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! Metric. The default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \param x Input point.
  //! \param radius Search radius.
  //! \code{.cpp}
  //! Scalar distance = -2.0;
  //! // E.g., L1: 2.0, L2Squared: 4.0
  //! Scalar metric_distance = kdtree.metric()(distance);
  //! std::vector<Neighbor<Index, Scalar>> n;
  //! tree.SearchRadius(p, metric_distance, &n);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& x,
      Scalar const radius,
      std::vector<NeighborType>* n,
      bool const sort = false) const {
    internal::SearchRadius<NeighborType> v(radius, n);
    SearchNearest(root_, x, &v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p x,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals Neighbor<Index, Scalar>.
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
  //! depend on the Metric. The default L2Squared calculates squared
  //! distances. Using this metric, the input error ratio should be the squared
  //! error ratio and the output distances will be squared distances scaled by
  //! the inverse error ratio.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! Scalar max_error = Scalar(0.15);
  //! Scalar e = tree.metric()(Scalar(1.0) + max_error);
  //! std::vector<Neighbor<Index, Scalar>> knn(k);
  //! tree.SearchAknn(p, e, knn.begin(), knn.end());
  //! // Optionally scale back to the actual metric distance.
  //! for (auto& nn : knn) { nn.second *= e; }
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchAknn(
      P const& x,
      Scalar const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>::value,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNearest(root_, x, &v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchAknn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchAknn(
      P const& x,
      Index const k,
      Scalar const e,
      std::vector<NeighborType>* knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn->resize(std::min(k, Traits::SpaceNpts(points_)));
    SearchAknn(x, e, knn->begin(), knn->end());
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
  inline Space const& points() const { return points_; }

  //! \brief Metric used for search queries.
  inline Metric const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static KdTree Load(Space points, std::string const& filename) {
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
  static KdTree Load(Space points, std::iostream* stream) {
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
  KdTree(Space points, internal::Stream* stream)
      : points_(std::move(points)),
        metric_(),
        nodes_(internal::MaxNodesFromPoints(Traits::SpaceNpts(points_))),
        indices_(Traits::SpaceNpts(points_)),
        root_(Load(stream)) {}

  inline void CalculateBoundingBox(Sequence* p_min, Sequence* p_max) {
    Sequence& min = *p_min;
    Sequence& max = *p_max;
    min.Fill(Traits::SpaceSdim(points_), std::numeric_limits<Scalar>::max());
    max.Fill(Traits::SpaceSdim(points_), std::numeric_limits<Scalar>::lowest());

    for (Index j = 0; j < Traits::SpaceNpts(points_); ++j) {
      Scalar const* const p = Traits::PointCoords(Traits::PointAt(points_, j));
      for (int i = 0;
           i < internal::Dimension<Dim>::Dim(Traits::SpaceSdim(points_));
           ++i) {
        Scalar const v = p[i];
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
    assert(Traits::SpaceNpts(points_) > 0);
    assert(max_leaf_size > 0);

    std::iota(indices_.begin(), indices_.end(), 0);

    CalculateBoundingBox(&root_box_min_, &root_box_max_);

    Splitter splitter(points_, &indices_);
    return Builder{max_leaf_size, splitter, &nodes_}.SplitIndices(
        0,
        0,
        Traits::SpaceNpts(points_),
        Sequence(root_box_min_),
        Sequence(root_box_max_));
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node .
  template <typename P, typename V>
  inline void SearchNearest(
      Node const* const node, P const& x, V* visitor) const {
    if (node->IsLeaf()) {
      for (Index i = node->data.leaf.begin_idx; i < node->data.leaf.end_idx;
           ++i) {
        Scalar const d = metric_(x, Traits::PointAt(points_, indices_[i]));
        if (visitor->max() > d) {
          (*visitor)(indices_[i], d);
        }
      }
    } else {
      // Go left or right and then check if we should still go down the other
      // side based on the current minimum distance.
      Scalar const v = x(node->data.branch.split_dim);
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

      SearchNearest(node_1st, x, visitor);
      if (visitor->max() >= metric_(node->data.branch.split_val, v)) {
        SearchNearest(node_2nd, x, visitor);
      }
    }
  }

  //! Checks if \p x is contained in the box defined by \p min and \p max. A
  //! point on the edge considered inside the box.
  template <typename P0, typename P1>
  inline bool PointInBox(P0 const& x, P1 const& min, P1 const& max) const {
    for (int i = 0;
         i < internal::Dimension<Dim>::Dim(Traits::SpaceSdim(points_));
         ++i) {
      if (min(i) > x(i) || max(i) < x(i)) {
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
        if (PointInBox(Traits::PointAt(points_, idx), rng_min, rng_max)) {
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
    Node* node = nodes_.Allocate();
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
  Space points_;
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
