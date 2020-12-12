#pragma once

template <typename Tree>
using TreeIndexType = typename std::remove_reference<decltype(
    std::declval<Tree>().points())>::type::Index;

template <typename Tree>
using TreeScalarType = typename std::remove_reference<decltype(
    std::declval<Tree>().points())>::type::Scalar;

template <typename Tree>
using TreePointsType = typename std::remove_reference<decltype(
    std::declval<Tree>().points())>::type;

template <
    typename P,
    typename Points,
    typename Index,
    typename Scalar,
    typename Metric>
void SearchKnn(
    P const& p,
    Points const& points,
    Index const k,
    Metric const& metric,
    std::vector<std::pair<Index, Scalar>>* knn) {
  Index const npts = points.npts();
  knn->resize(static_cast<std::size_t>(npts));
  for (Index i = 0; i < npts; ++i) {
    (*knn)[i] = {i, metric(p, points(i))};
  }

  Index const max_k = std::min(k, npts);
  std::nth_element(
      knn->begin(),
      knn->begin() + (max_k - 1),
      knn->end(),
      pico_tree::internal::NeighborComparator<Index, Scalar>());
  knn->resize(static_cast<std::size_t>(max_k));
  std::sort(
      knn->begin(),
      knn->end(),
      pico_tree::internal::NeighborComparator<Index, Scalar>());
}

// TODO Perhaps do something more friendly for exposing types.
// TODO Some traits.
template <typename Tree>
void TestBox(
    Tree const& tree,
    TreeScalarType<Tree> const min_v,
    TreeScalarType<Tree> const max_v) {
  using PointsX = TreePointsType<Tree>;
  using PointX = typename PointsX::Point;
  using Index = typename PointsX::Index;

  auto const& points = tree.points();

  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;
  tree.SearchBox(min, max, &idxs);

  for (auto j : idxs) {
    for (int d = 0; d < PointX::Dim; ++d) {
      EXPECT_GE(points(j)(d), min_v);
      EXPECT_LE(points(j)(d), max_v);
    }
  }

  std::size_t count = 0;

  for (Index j = 0; j < points.npts(); ++j) {
    bool contained = true;

    for (int d = 0; d < PointX::Dim; ++d) {
      if ((points(j)(d) < min_v) || (points(j)(d) > max_v)) {
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
void TestRadius(Tree const& tree, TreeScalarType<Tree> const radius) {
  using PointsX = TreePointsType<Tree>;
  using PointX = typename PointsX::Point;
  using Index = typename PointsX::Index;
  using Scalar = typename PointsX::Scalar;

  auto const& points = tree.points();

  Index idx = tree.points().npts() / 2;
  PointX p;

  for (Index d = 0; d < PointX::Dim; ++d) {
    p(d) = points(idx)(d);
  }

  auto const& metric = tree.metric();
  Scalar const lp_radius = metric(radius);
  std::vector<std::pair<Index, Scalar>> results;
  tree.SearchRadius(p, lp_radius, &results);

  for (auto const& r : results) {
    EXPECT_LE(metric(p, points(r.first)), lp_radius);
    EXPECT_EQ(metric(p, points(r.first)), r.second);
  }

  std::size_t count = 0;

  for (Index j = 0; j < points.npts(); ++j) {
    if (metric(p, points(j)) <= lp_radius) {
      count++;
    }
  }

  EXPECT_EQ(count, results.size());
}

template <typename Tree>
void TestKnn(Tree const& tree, TreeIndexType<Tree> const k) {
  using PointsX = TreePointsType<Tree>;
  using PointX = typename PointsX::Point;
  using Index = typename PointsX::Index;
  using Scalar = typename PointsX::Scalar;

  auto const& points = tree.points();

  Index idx = tree.points().npts() / 2;
  PointX p;

  for (Index d = 0; d < PointX::Dim; ++d) {
    p(d) = points(idx)(d);
  }

  Scalar ratio = tree.metric()(1.5);

  std::vector<std::pair<Index, Scalar>> results_exact;
  std::vector<std::pair<Index, Scalar>> results_apprx;
  tree.SearchKnn(p, k, &results_exact);
  tree.SearchAknn(p, k, ratio, &results_apprx);

  std::vector<std::pair<Index, Scalar>> compare;
  SearchKnn(p, points, k, tree.metric(), &compare);

  ASSERT_EQ(compare.size(), results_exact.size());
  for (std::size_t i = 0; i < compare.size(); ++i) {
    // Index is not tested in case it happens points have an equal distance.
    // TODO Would be nicer to test indices too.
    EXPECT_FLOAT_EQ(results_exact[i].second, compare[i].second);

    // Because results_apprx[i] is already scaled: approx = approx / ratio, the
    // check below is the same as: approx <= exact * ratio
    EXPECT_PRED_FORMAT2(
        testing::FloatLE, results_apprx[i].second, results_exact[i].second);
  }
}
