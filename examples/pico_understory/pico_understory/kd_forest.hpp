#pragma once

#include "pico_tree/internal/point_wrapper.hpp"
#include "pico_tree/internal/search_visitor.hpp"
#include "pico_tree/internal/space_wrapper.hpp"
#include "pico_tree/metric.hpp"
#include "pico_understory/internal/kd_tree_priority_search.hpp"
#include "pico_understory/internal/rkd_tree_builder.hpp"

namespace pico_tree {

template <
    typename Space_,
    typename Metric_ = L2Squared,
    SplittingRule SplittingRule_ = SplittingRule::kSlidingMidpoint,
    typename Index_ = int>
class KdForest {
  using SpaceWrapperType = internal::SpaceWrapper<Space_>;
  using NodeType = internal::
      KdTreeNodeTopological<Index_, typename SpaceWrapperType::ScalarType>;
  using BuildRKdTreeType =
      internal::BuildRKdTree<NodeType, SpaceWrapperType::Dim, SplittingRule_>;
  using RKdTreeDataType = typename BuildRKdTreeType::RKdTreeDataType;

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

  KdForest(SpaceType space, SizeType max_leaf_size, SizeType forest_size)
      : space_(std::move(space)),
        metric_(),
        data_(BuildRKdTreeType()(
            SpaceWrapperType(space_), max_leaf_size, forest_size)) {}

  //! \brief The KdForest cannot be copied.
  //! \details The KdForest uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  KdForest(KdForest const&) = delete;

  //! \brief Move constructor of the KdForest.
  KdForest(KdForest&&) = default;

  //! \brief KdForest copy assignment.
  KdForest& operator=(KdForest const& other) = delete;

  //! \brief KdForest move assignment.
  KdForest& operator=(KdForest&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  template <typename P, typename V>
  inline void SearchNearest(
      P const& x, SizeType max_leaves_visited, V& visitor) const {
    internal::PointWrapper<P> p(x);
    SearchNearest(p, max_leaves_visited, visitor, typename Metric_::SpaceTag());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default L2Squared results in a squared distance.
  template <typename P>
  inline void SearchNn(
      P const& x, SizeType max_leaves_visited, NeighborType& nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(x, max_leaves_visited, v);
  }

 private:
  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void SearchNearest(
      PointWrapper_ point,
      SizeType max_leaves_visited,
      Visitor_& visitor,
      EuclideanSpaceTag) const {
    // Range based for loop (rightfully) results in a warning that shouldn't be
    // needed if the user creates the forest with at least a single tree.
    for (std::size_t i = 0; i < data_.size(); ++i) {
      auto p = data_[i].RotatePoint(point);
      using PointWrapperType = internal::PointWrapper<decltype(p)>;
      PointWrapperType point_wrapper(p);
      internal::PrioritySearchNearestEuclidean<
          typename RKdTreeDataType::SpaceWrapperType,
          Metric_,
          PointWrapperType,
          Visitor_,
          IndexType>(
          typename RKdTreeDataType::SpaceWrapperType(data_[i].space),
          metric_,
          data_[i].tree.indices,
          point_wrapper,
          max_leaves_visited,
          visitor)(data_[i].tree.root_node);
    }
  }

  //! \brief Point set used for querying point data.
  SpaceType space_;
  //! \brief Metric used for comparing distances.
  MetricType metric_;
  //! \brief Data structure of the KdTree.
  std::vector<RKdTreeDataType> data_;
};

template <typename Space_>
KdForest(Space_, Size)
    -> KdForest<Space_, L2Squared, SplittingRule::kSlidingMidpoint, int>;

template <
    typename Metric_ = L2Squared,
    SplittingRule SplittingRule_ = SplittingRule::kSlidingMidpoint,
    typename Index_ = int,
    typename Space_>
auto MakeKdForest(Space_&& space, Size max_leaf_size, Size forest_size) {
  return KdForest<std::decay_t<Space_>, Metric_, SplittingRule_, Index_>(
      std::forward<Space_>(space), max_leaf_size, forest_size);
}

}  // namespace pico_tree
