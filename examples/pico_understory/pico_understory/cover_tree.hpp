#pragma once

#include <pico_tree/internal/point_wrapper.hpp>
#include <pico_tree/internal/search_visitor.hpp>
#include <pico_tree/internal/space_wrapper.hpp>

// Use this define to enable a simplified version of the nearest ancestor tree
// or disable it to use the regular one from "Faster Cover Trees".
//
// Testing seems to indicate that when the dataset has some structure (not
// random), the "simplified" version is faster.
// 1) Building: The performance difference can be large for a low leveling base,
// e.g., 1.3. The difference in building time becomes smaller when the leveling
// base increases.
// 2) Queries: Slower compared to the regular nearest ancestor tree for a low
// leveling base, but faster for a higher one.
//
// It seems that with structured data, both tree building and querying becomes
// faster when increasing the leveling base. Both times decreasing steadily when
// increasing the base. Here a value of 2.0 is the fastest.
//
// For random data, build and query times seem to be all over the place when
// steadily increasing the base. A value of 1.3 generally seems the fastest, but
// none of the values for this hyper parameter inspire trust.
#define SIMPLIFIED_NEAREST_ANCESTOR_COVER_TREE

#include "internal/cover_tree_builder.hpp"
#include "internal/cover_tree_data.hpp"
#include "internal/cover_tree_node.hpp"
#include "internal/cover_tree_search.hpp"
#include "metric.hpp"

namespace pico_tree {

template <typename Space_, typename Metric_ = L2, typename Index_ = int>
class CoverTree {
 private:
  using Index = Index_;
  using Space = Space_;
  using SpaceWrapperType = internal::SpaceWrapper<Space>;
  using Scalar = typename SpaceWrapperType::ScalarType;
  using Node = internal::CoverTreeNode<Index, Scalar>;
  using BuildCoverTreeType =
      internal::BuildCoverTree<SpaceWrapperType, Metric_, Index_>;
  using CoverTreeDataType = typename BuildCoverTreeType::CoverTreeDataType;

 public:
  //! \brief Index type.
  using IndexType = Index;
  //! \brief Scalar type.
  using ScalarType = Scalar;
  //! \brief CoverTree dimension. It equals pico_tree::kDynamicSize in case Dim
  //! is only known at run-time.
  static constexpr int Dim = SpaceWrapperType::Dim;
  //! \brief Point set or adaptor type.
  using SpaceType = Space;
  //! \brief The metric used for various searches.
  using MetricType = Metric_;
  //! \brief Neighbor type of various search resuls.
  using NeighborType = Neighbor<Index, Scalar>;

 public:
  //! \brief The CoverTree cannot be copied.
  //! \details The CoverTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy. For now we are not interested in
  //! providing a deep copy.
  //! \private
  CoverTree(CoverTree const&) = delete;

  //! \brief Move constructor of the CoverTree.
  //! \details The move constructor is not implicitly created because of the
  //! deleted copy constructor.
  //! \private
  CoverTree(CoverTree&&) = default;

  //! \brief Creates a CoverTree given \p points and a leveling \p base.
  CoverTree(Space space, Scalar base)
      : space_(std::move(space)),
        metric_(),
        data_(BuildCoverTreeType()(SpaceWrapperType(space_), metric_, base)) {}

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default distance metric equals L2.
  template <typename P>
  inline void SearchNn(P const& x, NeighborType& nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(data_.root_node, x, v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<Index, Scalar>.
  //! \details Interpretation of the output distances depend on the Metric. The
  //! default L2 metric results in Euclidean distances.
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
    SearchNearest(data_.root_node, x, v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void SearchKnn(P
  //! const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x, Size const k, std::vector<NeighborType>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, SpaceWrapperType(space_).size()));
    SearchKnn(x, knn.begin(), knn.end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  //! \details Interpretation of the in and output distances depend on the
  //! Metric. The default L2 results in squared distances.
  //! \tparam P Point type.
  //! \param x Input point.
  //! \param radius Search radius.
  //! \param n Output points.
  //! \param sort If true, the result set is sorted from closest to farthest
  //! distance with respect to the query point.
  template <typename P>
  inline void SearchRadius(
      P const& x,
      Scalar const radius,
      std::vector<NeighborType>& n,
      bool const sort = false) const {
    internal::SearchRadius<NeighborType> v(radius, n);
    SearchNearest(data_.root_node, x, v);

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
  //! Interpretation of both the input error ratio and output distances depend
  //! on the Metric. The default L2 metric calculates Euclidean distances.
  //!
  //! Example:
  //! \code{.cpp}
  //! // A max error of 15%. I.e. max 15% farther away from the true nn.
  //! Scalar max_error = Scalar(0.15);
  //! Scalar e = Scalar(1.0) + max_error;
  //! std::vector<Neighbor<Index, Scalar>> knn(k);
  //! tree.SearchKnn(x, e, knn.begin(), knn.end());
  //! \endcode
  template <typename P, typename RandomAccessIterator>
  inline void SearchKnn(
      P const& x,
      Scalar const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            NeighborType>,
        "SEARCH_ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_INDEX_SCALAR");

    internal::SearchAknn<RandomAccessIterator> v(e, begin, end);
    SearchNearest(data_.root_node, x, v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  //! \tparam P Point type.
  //! \see template <typename P, typename RandomAccessIterator> void
  //! SearchKnn(P const&, RandomAccessIterator, RandomAccessIterator) const
  template <typename P>
  inline void SearchKnn(
      P const& x,
      Size const k,
      Scalar const e,
      std::vector<NeighborType>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, SpaceWrapperType(space_).size()));
    SearchKnn(x, e, knn.begin(), knn.end());
  }

  //! \brief Point set used by the tree.
  inline Space const& points() const { return space_; }

  //! \brief Metric used for search queries.
  inline MetricType const& metric() const { return metric_; }

 private:
  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node .
  template <typename P, typename Visitor_>
  inline void SearchNearest(
      Node const* const node, P const& x, Visitor_& visitor) const {
    internal::PointWrapper<P> p(x);
    SpaceWrapperType space(space_);
    internal::SearchNearestMetric<
        SpaceWrapperType,
        MetricType,
        internal::PointWrapper<P>,
        Visitor_,
        IndexType>(space, metric_, p, visitor)(node);
  }

  //! Point set used for querying point data.
  SpaceType space_;
  //! Metric used for comparing distances.
  MetricType metric_;
  //! Data structure of the CoverTree.
  CoverTreeDataType data_;
};

}  // namespace pico_tree
