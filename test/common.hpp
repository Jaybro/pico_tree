#pragma once

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
    std::vector<pico_tree::Neighbor<Index, Scalar>>* knn) {
  Index const npts = points.npts();
  knn->resize(static_cast<std::size_t>(npts));
  for (Index i = 0; i < npts; ++i) {
    (*knn)[i] = {i, metric(p, points(i))};
  }

  Index const max_k = std::min(k, npts);
  std::nth_element(knn->begin(), knn->begin() + (max_k - 1), knn->end());
  knn->resize(static_cast<std::size_t>(max_k));
  std::sort(knn->begin(), knn->end());
}

template <typename Tree>
void TestBox(
    Tree const& tree,
    typename Tree::ScalarType const min_v,
    typename Tree::ScalarType const max_v) {
  using PointsX = typename Tree::PointsType;
  using PointX = typename PointsX::PointType;
  using Index = typename PointsX::IndexType;

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
void TestRadius(Tree const& tree, typename Tree::ScalarType const radius) {
  using PointsX = typename Tree::PointsType;
  using PointX = typename PointsX::PointType;
  using Index = typename PointsX::IndexType;
  using Scalar = typename PointsX::ScalarType;

  auto const& points = tree.points();

  Index idx = tree.points().npts() / 2;
  PointX p = points(idx);

  auto const& metric = tree.metric();
  Scalar const lp_radius = metric(radius);
  std::vector<pico_tree::Neighbor<Index, Scalar>> results;
  tree.SearchRadius(p, lp_radius, &results);

  for (auto const& r : results) {
    EXPECT_LE(metric(p, points(r.index)), lp_radius);
    EXPECT_EQ(metric(p, points(r.index)), r.distance);
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
void TestKnn(Tree const& tree, typename Tree::IndexType const k) {
  using PointsX = typename Tree::PointsType;
  using PointX = typename PointsX::PointType;
  using Index = typename PointsX::IndexType;
  using Scalar = typename PointsX::ScalarType;

  auto const& points = tree.points();

  Index idx = tree.points().npts() / 2;
  PointX p = points(idx);
  Scalar ratio = tree.metric()(1.5);

  std::vector<pico_tree::Neighbor<Index, Scalar>> results_exact;
  std::vector<pico_tree::Neighbor<Index, Scalar>> results_apprx;
  tree.SearchKnn(p, k, &results_exact);
  tree.SearchAknn(p, k, ratio, &results_apprx);

  std::vector<pico_tree::Neighbor<Index, Scalar>> compare;
  SearchKnn(p, points, k, tree.metric(), &compare);

  ASSERT_EQ(compare.size(), results_exact.size());
  for (std::size_t i = 0; i < compare.size(); ++i) {
    // Index is not tested in case it happens points have an equal distance.
    // TODO Would be nicer to test indices too.
    EXPECT_FLOAT_EQ(results_exact[i].distance, compare[i].distance);

    // Because results_apprx[i] is already scaled: approx = approx / ratio, the
    // check below is the same as: approx <= exact * ratio
    EXPECT_PRED_FORMAT2(
        testing::FloatLE, results_apprx[i].distance, results_exact[i].distance);
  }
}
