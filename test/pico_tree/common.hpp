#pragma once

#include <algorithm>
#include <pico_tree/core.hpp>
#include <pico_tree/internal/point_wrapper.hpp>
#include <pico_tree/internal/space_wrapper.hpp>
#include <pico_tree/map_traits.hpp>
#include <pico_tree/metric.hpp>

inline void float_eq(float val1, float val2) { EXPECT_FLOAT_EQ(val1, val2); }

inline void float_eq(double val1, double val2) { EXPECT_DOUBLE_EQ(val1, val2); }

inline void float_le(float val1, float val2) {
  EXPECT_PRED_FORMAT2(testing::FloatLE, val1, val2);
}

inline void float_le(double val1, double val2) {
  EXPECT_PRED_FORMAT2(testing::DoubleLE, val1, val2);
}

template <pico_tree::size_t Dim_, typename Space_, typename Scalar_>
void check_space_adaptor(
    Space_ const& space,
    pico_tree::size_t sdim,
    pico_tree::size_t npts,
    pico_tree::size_t point_index,
    Scalar_ const* point_data_ref) {
  using traits_type = pico_tree::space_traits<Space_>;

  static_assert(
      std::is_same_v<typename traits_type::space_type, Space_>,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(
      traits_type::dim == Dim_, "TRAITS_DIM_NOT_EQUAL_TO_EXPECTED_DIM");

  static_assert(
      std::is_same_v<typename traits_type::scalar_type, Scalar_>,
      "TRAITS_SCALAR_TYPE_INCORRECT");

  pico_tree::internal::space_wrapper<Space_> space_wrapper(space);

  EXPECT_EQ(sdim, space_wrapper.sdim());
  EXPECT_EQ(npts, space_wrapper.size());

  Scalar_ const* point_data_tst = space_wrapper[point_index];

  for (pico_tree::size_t i = 0; i < sdim; ++i) {
    float_eq(point_data_ref[i], point_data_tst[i]);
  }
}

template <typename Traits_, typename Point_, typename Index_, typename Metric_>
void search_knn(
    Point_ const& p,
    typename Traits_::space_type const& space,
    pico_tree::size_t const k,
    Metric_ const& metric,
    std::vector<pico_tree::neighbor<Index_, typename Traits_::scalar_type>>&
        knn) {
  //
  pico_tree::internal::space_wrapper<typename Traits_::space_type>
      space_wrapper(space);

  knn.resize(space_wrapper.size());
  for (pico_tree::size_t i = 0; i < space_wrapper.size(); ++i) {
    knn[i] = {
        static_cast<Index_>(i),
        metric(p.data(), p.data() + p.size(), space_wrapper[i])};
  }

  pico_tree::size_t max_k = std::min(k, space_wrapper.size());
  auto it_nth = knn.begin();
  std::advance(it_nth, max_k - 1);
  std::nth_element(knn.begin(), it_nth, knn.end());
  knn.resize(max_k);
  std::sort(knn.begin(), knn.end());
}

template <typename Tree_>
void test_box(
    Tree_ const& tree,
    typename Tree_::scalar_type const min_v,
    typename Tree_::scalar_type const max_v) {
  using point_type =
      typename pico_tree::space_traits<typename Tree_::space_type>::point_type;
  using scalar_type = typename Tree_::scalar_type;
  using index_type = typename Tree_::index_type;
  constexpr pico_tree::size_t dim = Tree_::dim;
  // TODO Because of the min_v and max_v arguments, test_box only works on so^n
  // topological metrics.
  using metric_type = typename Tree_::metric_type;
  using box_map_type = std::conditional_t<
      std::is_same_v<
          typename metric_type::space_category,
          pico_tree::euclidean_space_tag>,
      pico_tree::internal::box_map<scalar_type const, dim>,
      pico_tree::internal::metric_box_map<scalar_type const, dim, metric_type>>;

  pico_tree::internal::space_wrapper<typename Tree_::space_type> space(
      tree.space());

  point_type min, max;
  min.fill(min_v);
  max.fill(max_v);
  box_map_type box_map(
      pico_tree::internal::point_wrapper<point_type>(min).begin(),
      pico_tree::internal::point_wrapper<point_type>(max).begin(),
      space.sdim());

  std::vector<index_type> idxs;
  tree.search_box(min, max, idxs);

  std::size_t count = 0;

  for (auto j : idxs) {
    EXPECT_TRUE(box_map.contains(space[j]));
  }

  for (pico_tree::size_t j = 0; j < space.size(); ++j) {
    if (box_map.contains(space[j])) {
      count++;
    }
  }

  EXPECT_EQ(count, idxs.size());
}

template <typename Tree_>
void test_radius(Tree_ const& tree, typename Tree_::scalar_type const radius) {
  using index_type = typename Tree_::index_type;
  using scalar_type = typename Tree_::scalar_type;
  constexpr auto dim = Tree_::dim;

  pico_tree::internal::space_wrapper<typename Tree_::space_type> space(
      tree.space());
  auto p = pico_tree::point_map<scalar_type const, dim>(
      space[space.size() / 2], space.sdim());

  auto const& metric = tree.metric();
  scalar_type lp_radius = metric(radius);
  scalar_type lp_scale = metric(scalar_type(1.5));

  std::vector<pico_tree::neighbor<index_type, scalar_type>> results_exact;
  std::vector<pico_tree::neighbor<index_type, scalar_type>> results_apprx;
  tree.search_radius(p, lp_radius, results_exact);
  tree.search_radius(p, lp_radius, lp_scale, results_apprx);

  for (auto const& r : results_exact) {
    scalar_type d = metric(p.data(), p.data() + p.size(), space[r.index]);

    EXPECT_LE(d, lp_radius);
    EXPECT_EQ(d, r.distance);
  }

  for (auto const& r : results_apprx) {
    scalar_type d = metric(p.data(), p.data() + p.size(), space[r.index]);

    EXPECT_LE(d, lp_radius);
    float_eq(d, r.distance * lp_scale);
  }

  std::size_t count = 0;

  for (pico_tree::size_t j = 0; j < space.size(); ++j) {
    if (metric(p.data(), p.data() + p.size(), space[j]) <= lp_radius) {
      count++;
    }
  }

  EXPECT_EQ(count, results_exact.size());
  EXPECT_GE(count, results_apprx.size());
}

template <typename Tree_, typename Point_>
void test_knn(Tree_ const& tree, pico_tree::size_t const k, Point_ const& p) {
  using traits_type = pico_tree::space_traits<typename Tree_::space_type>;
  using index_type = typename Tree_::index_type;
  using scalar_type = typename Tree_::scalar_type;

  auto const space = tree.space();
  scalar_type lp_scale = tree.metric()(scalar_type(1.5));

  std::vector<pico_tree::neighbor<index_type, scalar_type>> results_exact;
  std::vector<pico_tree::neighbor<index_type, scalar_type>> results_apprx;
  tree.search_knn(p, k, results_exact);
  tree.search_knn(p, k, lp_scale, results_apprx);

  std::vector<pico_tree::neighbor<index_type, scalar_type>> compare;
  search_knn<traits_type>(p, space, k, tree.metric(), compare);

  ASSERT_EQ(compare.size(), results_exact.size());
  for (std::size_t i = 0; i < compare.size(); ++i) {
    // Index is not tested in case it happens points have an equal distance.
    // TODO Would be nicer to test indices too.
    float_eq(results_exact[i].distance, compare[i].distance);
    // Because results_apprx[i] is already scaled: approx = approx / lp_scale,
    // the check below is the same as: approx <= exact * lp_scale.
    float_le(results_apprx[i].distance, results_exact[i].distance);
  }
}

template <typename Tree_>
void test_knn(Tree_ const& tree, pico_tree::size_t const k) {
  using scalar_type = typename Tree_::scalar_type;
  constexpr auto dim = Tree_::dim;

  auto space = pico_tree::internal::space_wrapper<typename Tree_::space_type>(
      tree.space());
  auto p = pico_tree::point_map<scalar_type const, dim>(
      space[space.size() / 2], space.sdim());

  test_knn(tree, k, p);
}
