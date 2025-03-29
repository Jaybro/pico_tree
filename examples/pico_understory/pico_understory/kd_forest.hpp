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
    typename Metric_ = metric_l2_squared,
    typename Index_ = int>
class kd_forest {
  using space_wrapper_type = internal::space_wrapper<Space_>;
  // priority_search_nearest_euclidean only supports kd_tree_node_topological.
  using node_type = internal::kd_tree_node_topological<
      Index_,
      typename space_wrapper_type::scalar_type>;
  using rkd_tree_data_type =
      internal::rkd_tree_hh_data<node_type, space_wrapper_type::dim>;

 public:
  //! \brief Size type.
  using size_type = size_t;
  //! \brief Index type.
  using index_type = Index_;
  //! \brief Scalar type.
  using scalar_type = typename space_wrapper_type::scalar_type;
  //! \brief kd_tree dimension. It equals pico_tree::dynamic_extent in case
  //! dim is only known at run-time.
  static size_type constexpr dim = space_wrapper_type::dim;
  //! \brief Point set or adaptor type.
  using space_type = Space_;
  //! \brief The metric used for various searches.
  using metric_type = Metric_;
  //! \brief Neighbor type of various search resuls.
  using neighbor_type = neighbor<index_type, scalar_type>;

  kd_forest(space_type space, size_type max_leaf_size, size_type forest_size)
      : space_(std::move(space)),
        metric_(),
        data_(
            internal::build_rkd_tree<rkd_tree_data_type, dim>()(
                space_wrapper_type(space_),
                max_leaf_size_t(max_leaf_size),
                bounds_from_space,
                sliding_midpoint_max_side,
                forest_size)) {}

  //! \brief The kd_forest cannot be copied.
  //! \details The kd_forest uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy.
  kd_forest(kd_forest const&) = delete;

  //! \brief Move constructor of the kd_forest.
  kd_forest(kd_forest&&) = default;

  //! \brief kd_forest copy assignment.
  kd_forest& operator=(kd_forest const& other) = delete;

  //! \brief kd_forest move assignment.
  kd_forest& operator=(kd_forest&& other) = default;

  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor .
  template <typename P_, typename V_>
  inline void search_nearest(
      P_ const& x, size_type max_leaves_visited, V_& visitor) const {
    internal::point_wrapper<P_> p(x);
    search_nearest(
        p, max_leaves_visited, visitor, typename Metric_::space_category());
  }

  //! \brief Searches for the nearest neighbor of point \p x.
  //! \details Interpretation of the output distance depends on the Metric. The
  //! default metric_l2_squared results in a squared distance.
  template <typename P_>
  inline void search_nn(
      P_ const& x, size_type max_leaves_visited, neighbor_type& nn) const {
    internal::search_nn<neighbor_type> v(nn);
    search_nearest(x, max_leaves_visited, v);
  }

 private:
  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node.
  template <typename PointWrapper_, typename Visitor_>
  inline void search_nearest(
      PointWrapper_ point,
      size_type max_leaves_visited,
      Visitor_& visitor,
      euclidean_space_tag) const {
    // Range based for loop (rightfully) results in a warning that shouldn't be
    // needed if the user creates the forest with at least a single tree.
    for (std::size_t i = 0; i < data_.size(); ++i) {
      auto p = data_[i].rotate_point(point);
      using point_wrapper_type = internal::point_wrapper<decltype(p)>;
      point_wrapper_type point_wrapper(p);
      internal::priority_search_nearest_euclidean<
          typename rkd_tree_data_type::space_wrapper_type,
          Metric_,
          point_wrapper_type,
          Visitor_,
          index_type>(
          typename rkd_tree_data_type::space_wrapper_type(data_[i].space),
          metric_,
          data_[i].tree.indices,
          point_wrapper,
          max_leaves_visited,
          visitor)(data_[i].tree.root_node);
    }
  }

  //! \brief Point set used for querying point data.
  space_type space_;
  //! \brief Metric used for comparing distances.
  metric_type metric_;
  //! \brief Data structure of the kd_tree.
  std::vector<rkd_tree_data_type> data_;
};

template <typename Space_>
kd_forest(Space_, size_t, size_t) -> kd_forest<Space_, metric_l2_squared, int>;

template <
    typename Metric_ = metric_l2_squared,
    typename Index_ = int,
    typename Space_>
kd_forest<std::decay_t<Space_>, Metric_, Index_> make_kd_forest(
    Space_&& space, size_t max_leaf_size, size_t forest_size) {
  return kd_forest<std::decay_t<Space_>, Metric_, Index_>(
      std::forward<Space_>(space), max_leaf_size, forest_size);
}

}  // namespace pico_tree
