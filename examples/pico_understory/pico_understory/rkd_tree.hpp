#pragma once

#include <pico_tree/internal/point_wrapper.hpp>
#include <pico_tree/internal/space_wrapper.hpp>
#include <pico_tree/metric.hpp>

#include "pico_understory/internal/rkd_tree_builder.hpp"
#include "pico_understory/internal/rkd_tree_search.hpp"

namespace pico_tree {

template <
    typename Space_,
    typename Metric_ = L2Squared,
    SplittingRule SplittingRule_ = SplittingRule::kSlidingMidpoint,
    typename Index_ = int>
class RKdTree {
  using SpaceWrapperType = internal::SpaceWrapper<Space_>;
  //! \brief Node type based on Metric_::SpaceTag.
  using NodeType =
      typename internal::KdTreeSpaceTagTraits<typename Metric_::SpaceTag>::
          template NodeType<Index_, typename SpaceWrapperType::ScalarType>;
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

  RKdTree(SpaceType space, SizeType max_leaf_size, SizeType forest_size)
      : space_(std::move(space)),
        metric_(),
        data_(BuildRKdTreeType()(
            SpaceWrapperType(space_), max_leaf_size, forest_size)) {}

  //! \brief The RKdTree cannot be copied.
  //! \details The RKdTree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  RKdTree(RKdTree const&) = delete;

  //! \brief Move constructor of the RKdTree.
  RKdTree(RKdTree&&) = default;

  //! \brief RKdTree copy assignment.
  RKdTree& operator=(RKdTree const& other) = delete;

  //! \brief RKdTree move assignment.
  RKdTree& operator=(RKdTree&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  //! \see internal::SearchNn
  //! \see internal::SearchKnn
  //! \see internal::SearchRadius
  //! \see internal::SearchAknn
  template <typename P, typename V>
  inline void SearchNearest(P const& x, V& visitor) const {
    internal::PointWrapper<P> p(x);
    SearchNearest(p, visitor, typename Metric_::SpaceTag());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default L2Squared results in a squared distance.
  template <typename P>
  inline void SearchNn(P const& x, NeighborType& nn) const {
    internal::SearchNn<NeighborType> v(nn);
    SearchNearest(x, v);
  }

 private:
  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void SearchNearest(
      PointWrapper_ point, Visitor_& visitor, EuclideanSpaceTag) const {
    // Range based for loop (rightfully) results in a warning that shouldn't be
    // needed if the user creates the forest with at least a single tree.
    for (std::size_t i = 0; i < data_.size(); ++i) {
      auto p = data_[i].RotatePoint(point);
      using PointWrapperType = internal::PointWrapper<decltype(p)>;
      PointWrapperType w(p);
      internal::SearchNearestEuclideanDefeatist<
          typename RKdTreeDataType::SpaceWrapperType,
          Metric_,
          PointWrapperType,
          Visitor_,
          IndexType>(
          typename RKdTreeDataType::SpaceWrapperType(data_[i].space),
          metric_,
          data_[i].tree.indices,
          w,
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

}  // namespace pico_tree
