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

template <typename Space_, typename Metric_ = metric_l2, typename Index_ = int>
class cover_tree {
 private:
  using space_wrapper_type = internal::space_wrapper<Space_>;
  using node_type = internal::
      cover_tree_node<Index_, typename space_wrapper_type::scalar_type>;
  using build_cover_tree_type =
      internal::build_cover_tree<space_wrapper_type, Metric_, Index_>;
  using cover_tree_data_type =
      typename build_cover_tree_type::cover_tree_data_type;

 public:
  //! \brief Index type.
  using index_type = Index_;
  //! \brief scalar_type type.
  using scalar_type = typename space_wrapper_type::scalar_type;
  //! \brief cover_tree dimension. It equals pico_tree::dynamic_size in case
  //! dim is only known at run-time.
  static constexpr int dim = space_wrapper_type::dim;
  //! \brief Point set or adaptor type.
  using space_type = Space_;
  //! \brief The metric used for various searches.
  using metric_type = Metric_;
  //! \brief Neighbor type of various search resuls.
  using neighbor_type = neighbor<index_type, scalar_type>;

 public:
  //! \brief The cover_tree cannot be copied.
  //! \details The cover_tree uses pointers to nodes and copying pointers is not
  //! the same as creating a deep copy. For now we are not interested in
  //! providing a deep copy.
  //! \private
  cover_tree(cover_tree const&) = delete;

  //! \brief Move constructor of the cover_tree.
  //! \details The move constructor is not implicitly created because of the
  //! deleted copy constructor.
  //! \private
  cover_tree(cover_tree&&) = default;

  //! \brief Creates a cover_tree given \p points and a leveling \p base.
  cover_tree(space_type space, scalar_type base)
      : space_(std::move(space)),
        metric_(),
        data_(build_cover_tree_type()(
            space_wrapper_type(space_), metric_, base)) {}

  //! \brief Searches for the nearest neighbor of point \p x.
  template <typename P_>
  inline void search_nn(P_ const& x, neighbor_type& nn) const {
    internal::search_nn<neighbor_type> v(nn);
    search_nearest(data_.root_node, x, v);
  }

  //! \brief Searches for an approximate nearest neighbor of point \p x.
  template <typename P_>
  inline void search_nn(
      P_ const& x, scalar_type const e, neighbor_type& nn) const {
    internal::search_approximate_nn<neighbor_type> v(e, nn);
    search_nearest(data_.root_node, x, v);
  }

  //! \brief Searches for the k nearest neighbors of point \p x, where k equals
  //! std::distance(begin, end). It is expected that the value type of the
  //! iterator equals Neighbor<Index, scalar_type>.
  template <typename P_, typename RandomAccessIterator>
  inline void search_knn(
      P_ const& x, RandomAccessIterator begin, RandomAccessIterator end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            neighbor_type>,
        "ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_TYPE");

    internal::search_knn<RandomAccessIterator> v(begin, end);
    search_nearest(data_.root_node, x, v);
  }

  //! \brief Searches for the \p k nearest neighbors of point \p x and stores
  //! the results in output vector \p knn.
  template <typename P_>
  inline void search_knn(
      P_ const& x, size_t const k, std::vector<neighbor_type>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, space_wrapper_type(space_).size()));
    search_knn(x, knn.begin(), knn.end());
  }

  //! \brief Searches for the k approximate nearest neighbors of point \p x,
  //! where k equals std::distance(begin, end). It is expected that the value
  //! type of the iterator equals Neighbor<Index, scalar_type>.
  template <typename P_, typename RandomAccessIterator>
  inline void search_knn(
      P_ const& x,
      scalar_type const e,
      RandomAccessIterator begin,
      RandomAccessIterator end) const {
    static_assert(
        std::is_same_v<
            typename std::iterator_traits<RandomAccessIterator>::value_type,
            neighbor_type>,
        "ITERATOR_VALUE_TYPE_DOES_NOT_EQUAL_NEIGHBOR_TYPE");

    internal::search_approximate_knn<RandomAccessIterator> v(e, begin, end);
    search_nearest(data_.root_node, x, v);
  }

  //! \brief Searches for the \p k approximate nearest neighbors of point \p x
  //! and stores the results in output vector \p knn.
  template <typename P_>
  inline void search_knn(
      P_ const& x,
      size_t const k,
      scalar_type const e,
      std::vector<neighbor_type>& knn) const {
    // If it happens that the point set has less points than k we just return
    // all points in the set.
    knn.resize(std::min(k, space_wrapper_type(space_).size()));
    search_knn(x, e, knn.begin(), knn.end());
  }

  //! \brief Searches for all the neighbors of point \p x that are within radius
  //! \p radius and stores the results in output vector \p n.
  template <typename P_>
  inline void search_radius(
      P_ const& x,
      scalar_type const radius,
      std::vector<neighbor_type>& n,
      bool const sort = false) const {
    internal::search_radius<neighbor_type> v(radius, n);
    search_nearest(data_.root_node, x, v);

    if (sort) {
      v.sort();
    }
  }

  //! \brief Searches for the approximate neighbors of point \p x that are
  //! within radius \p radius and stores the results in output vector \p n.
  template <typename P_>
  inline void search_radius(
      P_ const& x,
      scalar_type const radius,
      scalar_type const e,
      std::vector<neighbor_type>& n,
      bool const sort = false) const {
    internal::search_approximate_radius<neighbor_type> v(e, radius, n);
    search_nearest(data_.root_node, x, v);

    if (sort) {
      v.sort();
    }
  }

  //! \brief Point set used by the tree.
  inline space_type const& points() const { return space_; }

  //! \brief Metric used for search queries.
  inline metric_type const& metric() const { return metric_; }

 private:
  //! \brief Returns the nearest neighbor (or neighbors) of point \p x depending
  //! on their selection by visitor \p visitor for node \p node .
  template <typename P_, typename Visitor_>
  inline void search_nearest(
      node_type const* const node, P_ const& x, Visitor_& visitor) const {
    internal::point_wrapper<P_> p(x);
    space_wrapper_type space(space_);
    internal::search_nearest_metric<
        space_wrapper_type,
        metric_type,
        internal::point_wrapper<P_>,
        Visitor_,
        index_type>(space, metric_, p, visitor)(node);
  }

  //! Point set used for querying point data.
  space_type space_;
  //! Metric used for comparing distances.
  metric_type metric_;
  //! Data structure of the cover_tree.
  cover_tree_data_type data_;
};

}  // namespace pico_tree
