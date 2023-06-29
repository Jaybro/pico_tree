#pragma once

#include <algorithm>
#include <pico_tree/core.hpp>
#include <pico_tree/internal/space_wrapper.hpp>
#include <pico_tree/map_traits.hpp>

inline void FloatEq(float val1, float val2) { EXPECT_FLOAT_EQ(val1, val2); }

inline void FloatEq(double val1, double val2) { EXPECT_DOUBLE_EQ(val1, val2); }

inline void FloatLe(float val1, float val2) {
  EXPECT_PRED_FORMAT2(testing::FloatLE, val1, val2);
}

inline void FloatLe(double val1, double val2) {
  EXPECT_PRED_FORMAT2(testing::DoubleLE, val1, val2);
}

template <pico_tree::Size Dim, typename Space_, typename Scalar_>
void CheckSpaceAdaptor(
    Space_ const& space,
    pico_tree::Size sdim,
    pico_tree::Size npts,
    pico_tree::Size point_index,
    Scalar_ const* point_data_ref) {
  using Traits = pico_tree::SpaceTraits<Space_>;

  static_assert(
      std::is_same_v<typename Traits::SpaceType, Space_>,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(Traits::Dim == Dim, "TRAITS_DIM_NOT_EQUAL_TO_EXPECTED_DIM");

  static_assert(
      std::is_same_v<typename Traits::ScalarType, Scalar_>,
      "TRAITS_SCALAR_TYPE_INCORRECT");

  pico_tree::internal::SpaceWrapper<Space_> space_wrapper(space);

  EXPECT_EQ(sdim, space_wrapper.sdim());
  EXPECT_EQ(npts, space_wrapper.size());

  Scalar_ const* point_data_tst = space_wrapper[point_index];

  for (pico_tree::Size i = 0; i < sdim; ++i) {
    FloatEq(point_data_ref[i], point_data_tst[i]);
  }
}

template <typename Traits, typename Point, typename Index, typename Metric>
void SearchKnn(
    Point const& p,
    typename Traits::SpaceType const& space,
    pico_tree::Size const k,
    Metric const& metric,
    std::vector<pico_tree::Neighbor<Index, typename Traits::ScalarType>>* knn) {
  //
  pico_tree::internal::SpaceWrapper<typename Traits::SpaceType> space_wrapper(
      space);

  knn->resize(space_wrapper.size());
  for (pico_tree::Size i = 0; i < space_wrapper.size(); ++i) {
    (*knn)[i] = {
        static_cast<Index>(i),
        metric(p.data(), p.data() + p.size(), space_wrapper[i])};
  }

  pico_tree::Size const max_k = std::min(k, space_wrapper.size());
  std::nth_element(knn->begin(), knn->begin() + (max_k - 1), knn->end());
  knn->resize(max_k);
  std::sort(knn->begin(), knn->end());
}

template <typename Tree>
void TestBox(
    Tree const& tree,
    typename Tree::ScalarType const min_v,
    typename Tree::ScalarType const max_v) {
  using PointX =
      typename pico_tree::SpaceTraits<typename Tree::SpaceType>::PointType;
  using Index = typename Tree::IndexType;

  pico_tree::internal::SpaceWrapper<typename Tree::SpaceType> points(
      tree.points());

  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;
  tree.SearchBox(min, max, idxs);

  for (auto j : idxs) {
    for (pico_tree::Size d = 0; d < points.sdim(); ++d) {
      auto v = points[j][d];
      EXPECT_GE(v, min_v);
      EXPECT_LE(v, max_v);
    }
  }

  std::size_t count = 0;

  for (pico_tree::Size j = 0; j < points.size(); ++j) {
    bool contained = true;

    for (pico_tree::Size d = 0; d < points.sdim(); ++d) {
      auto v = points[j][d];
      if ((v < min_v) || (v > max_v)) {
        contained = false;
        break;
      }
    }

    if (contained) {
      count++;
    }
  }

  EXPECT_EQ(count, idxs.size());
}

template <typename Tree>
void TestRadius(Tree const& tree, typename Tree::ScalarType const radius) {
  using Index = typename Tree::IndexType;
  using Scalar = typename Tree::ScalarType;
  constexpr auto Dim = Tree::Dim;

  pico_tree::internal::SpaceWrapper<typename Tree::SpaceType> points(
      tree.points());
  auto p = pico_tree::PointMap<Scalar const, Dim>(
      points[points.size() / 2], points.sdim());

  auto const& metric = tree.metric();
  Scalar const lp_radius = metric(radius);
  std::vector<pico_tree::Neighbor<Index, Scalar>> results;
  tree.SearchRadius(p, lp_radius, results);

  for (auto const& r : results) {
    Scalar d = metric(p.data(), p.data() + p.size(), points[r.index]);

    EXPECT_LE(d, lp_radius);
    EXPECT_EQ(d, r.distance);
  }

  std::size_t count = 0;

  for (pico_tree::Size j = 0; j < points.size(); ++j) {
    if (metric(p.data(), p.data() + p.size(), points[j]) <= lp_radius) {
      count++;
    }
  }

  EXPECT_EQ(count, results.size());
}

template <typename Tree, typename Point>
void TestKnn(Tree const& tree, pico_tree::Size const k, Point const& p) {
  using TraitsX = pico_tree::SpaceTraits<typename Tree::SpaceType>;
  using Index = typename Tree::IndexType;
  using Scalar = typename Tree::ScalarType;

  // The data doesn't have to be by reference_wrapper, but that prevents a copy.
  auto const points = tree.points();
  Scalar ratio = tree.metric()(Scalar(1.5));

  std::vector<pico_tree::Neighbor<Index, Scalar>> results_exact;
  std::vector<pico_tree::Neighbor<Index, Scalar>> results_apprx;
  tree.SearchKnn(p, k, results_exact);
  tree.SearchKnn(p, k, ratio, results_apprx);

  std::vector<pico_tree::Neighbor<Index, Scalar>> compare;
  SearchKnn<TraitsX>(p, points, k, tree.metric(), &compare);

  ASSERT_EQ(compare.size(), results_exact.size());
  for (std::size_t i = 0; i < compare.size(); ++i) {
    // Index is not tested in case it happens points have an equal distance.
    // TODO Would be nicer to test indices too.
    FloatEq(results_exact[i].distance, compare[i].distance);
    // Because results_apprx[i] is already scaled: approx = approx / ratio,
    // the check below is the same as: approx <= exact * ratio
    FloatLe(results_apprx[i].distance, results_exact[i].distance);
  }
}

template <typename Tree>
void TestKnn(Tree const& tree, typename Tree::IndexType const k) {
  using ScalarType = typename Tree::ScalarType;
  constexpr auto Dim = Tree::Dim;

  auto points = pico_tree::internal::SpaceWrapper<typename Tree::SpaceType>(
      tree.points());
  auto p = pico_tree::PointMap<ScalarType const, Dim>(
      points[points.size() / 2], points.sdim());

  TestKnn(tree, k, p);
}
