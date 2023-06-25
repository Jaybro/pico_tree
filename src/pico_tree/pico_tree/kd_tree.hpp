#pragma once

#include "pico_tree/internal/box.hpp"
#include "pico_tree/internal/kd_tree_builder.hpp"
#include "pico_tree/internal/kd_tree_search.hpp"
#include "pico_tree/internal/point_wrapper.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/space_wrapper.hpp"

namespace pico_tree {

//! \brief A KdTree is a binary tree that partitions space using hyper planes.
//! \details https://en.wikipedia.org/wiki/K-d_tree
template <
    typename Space_,
    typename Metric_ = L2Squared,
    SplittingRule SplittingRule_ = SplittingRule::kSlidingMidpoint,
    typename Index_ = int>
class KdTree {
  using SpaceWrapperType = internal::SpaceWrapper<Space_>;
  using BuildKdTreeType =
      internal::BuildKdTree<SpaceWrapperType, Metric_, SplittingRule_, Index_>;
  using KdTreeDataType = typename BuildKdTreeType::KdTreeDataType;
  using NodeType = typename KdTreeDataType::NodeType;

 public:
  //! \brief Size type.
  using SizeType = Size;
  //! \brief Index type.
  using IndexType = Index_;
  //! \brief Scalar type.
  using ScalarType = typename SpaceWrapperType::ScalarType;
  //! \brief KdTree dimension. It equals pico_tree::kDynamicSize in case Dim is
  //! only known at run-time.
  static SizeType constexpr Dim = SpaceWrapperType::Dim;
  //! \brief Point set or adaptor type.
  using SpaceType = Space_;
  //! \brief The metric used for various searches.
  using MetricType = Metric_;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<IndexType, ScalarType>;

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
  KdTree(SpaceType space, SizeType max_leaf_size)
      : space_(std::move(space)),
        metric_(),
        data_(BuildKdTreeType()(SpaceWrapperType(space_), max_leaf_size)) {}

  //! \brief The KdTree cannot be copied.
  //! \details The KdTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  KdTree(KdTree const&) = delete;

  //! \brief Move constructor of the KdTree.
  KdTree(KdTree&&) = default;

  //! \brief KdTree copy assignment.
  KdTree& operator=(KdTree const& other) = delete;

  //! \brief KdTree move assignment.
  KdTree& operator=(KdTree&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNearest(P const& x, V& visitor) const {
    internal::PointWrapper<P> p(x);
    SearchNearest(data_.root_node, p, visitor, typename Metric_::SpaceTag());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default L2Squared results in a squared distance.
  template <typename P>
  inline void SearchNn(P const& x, NeighborType& nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(x, v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<IndexType, ScalarType>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \tparam RandomAccessIterator Iterator type.
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& x, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchKnn<RandomAccessIterator> v(begin, end);
    SearchNearest(x, v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x, SizeType const k, std::vector<NeighborType>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, SpaceWrapperType(space_).size()));
    SearchKnn(x, knn.begin(), knn.end());
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p x,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals Neighbor<IndexType, ScalarType>.
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
  //! ScalarType max_error = ScalarType(0.15);
  //! ScalarType e = tree.metric()(ScalarType(1.0) + max_error);
  //! std::vector<Neighbor<IndexType, ScalarType>> knn(k);
  //! tree.SearchKnn(p, e, knn.begin(), knn.end());
  //! // Optionally scale back to the actual metric distance.
  //! for (auto& nn : knn) { nn.second *= e; }
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& x,
      ScalarType const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNearest(x, v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchKnn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x,
      SizeType const k,
      ScalarType const e,
      std::vector<NeighborType>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, SpaceWrapperType(space_).size()));
    SearchKnn(x, e, knn.begin(), knn.end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! Metric. The default L2Squared results in squared distances.
  //! \tparam P Point type.
  //! \param x Input point.
  //! \param radius Search radius.
  //! \code{.cpp}
  //! ScalarType distance = -2.0;
  //! // E.g., L1: 2.0, L2Squared: 4.0
  //! ScalarType metric_distance = kdtree.metric()(distance);
  //! std::vector<Neighbor<IndexType, ScalarType>> n;
  //! tree.SearchRadius(p, metric_distance, n);
  //! \endcode
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& x,
      ScalarType const radius,
      std::vector<NeighborType>& n,
      bool const sort = false) const {
    internal::SearchRadius<NeighborType> v(radius, n);
    SearchNearest(x, v);

    if (sort) {
      v.Sort();
    }
  }

  //! \brief Returns all points within the box defined by \p min and \p max.
  //! Query time is bounded by O(n^(1-1/Dim)+k).
  //! \tparam P Point type.
  template <typename P>
  inline void SearchBox(
      P const& min, P const& max, std::vector<IndexType>& idxs) const {
    idxs.clear();
    SpaceWrapperType space(space_);
    // Note that it's never checked if the bounding box intersects at all. For
    // now it is assumed that this check is not worth it: If there is any
    // overlap then the search is slower. So unless many queries don't intersect
    // there is no point in adding it.
    internal::SearchBoxEuclidean<SpaceWrapperType, Metric_, IndexType>(
        space,
        metric_,
        data_.indices,
        data_.root_box,
        internal::BoxMap<ScalarType const, Dim>(
            internal::PointWrapper<P>(min).begin(),
            internal::PointWrapper<P>(max).begin(),
            space.sdim()),
        idxs)(data_.root_node);
  }

  //! \brief Point set used by the tree.
  inline SpaceType const& points() const { return space_; }

  //! \brief Metric used for search queries.
  inline MetricType const& metric() const { return metric_; }

  //! \brief Loads the tree in binary from file.
  static KdTree Load(SpaceType points, std::string const& filename) {
    std::fstream stream =
        internal::OpenStream(filename, std::ios::in | std::ios::binary);
    return Load(std::move(points), &stream);
  }

  //! \brief Loads the tree in binary from \p stream .
  //! \details This is considered a convinience function to be able to save and
  //! load a KdTree on a single machine.
  //! \li Does not take memory endianness into account.
  //! \li Does not check if the stored tree structure is valid for the given
  //! point set.
  //! \li Does not check if the stored tree structure is valid for the given
  //! template arguments.
  static KdTree Load(SpaceType points, std::iostream* stream) {
    internal::Stream s(*stream);
    return KdTree(std::move(points), s);
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
    internal::Stream s(*stream);
    KdTreeDataType::Save(tree.data_, s);
  }

 private:
  //! \brief Constructs a KdTree by reading its indexing and leaf information
  //! from a Stream.
  KdTree(SpaceType space, internal::Stream& stream)
      : space_(std::move(space)),
        metric_(),
        data_(KdTreeDataType::Load(stream)) {}

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void SearchNearest(
      NodeType const* const node,
      PointWrapper_ const& point,
      Visitor_& visitor,
      EuclideanSpaceTag) const {
    SpaceWrapperType space(space_);
    internal::SearchNearestEuclidean<
        SpaceWrapperType,
        Metric_,
        PointWrapper_,
        Visitor_,
        IndexType>(space, metric_, data_.indices, point, visitor)(node);
  }

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void SearchNearest(
      NodeType const* const node,
      PointWrapper_ const& point,
      Visitor_& visitor,
      TopologicalSpaceTag) const {
    SpaceWrapperType space(space_);
    internal::SearchNearestTopological<
        SpaceWrapperType,
        Metric_,
        PointWrapper_,
        Visitor_,
        IndexType>(space, metric_, data_.indices, point, visitor)(node);
  }

  //! \brief Point set used for querying point data.
  SpaceType space_;
  //! \brief Metric used for comparing distances.
  MetricType metric_;
  //! \brief Data structure of the KdTree.
  KdTreeDataType data_;
};

}  // namespace pico_tree
