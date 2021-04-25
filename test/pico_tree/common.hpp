#pragma once

inline void FloatEq(float val1, float val2) { EXPECT_FLOAT_EQ(val1, val2); }

inline void FloatEq(double val1, double val2) { EXPECT_DOUBLE_EQ(val1, val2); }

inline void FloatLe(float val1, float val2) {
  EXPECT_PRED_FORMAT2(testing::FloatLE, val1, val2);
}

inline void FloatLe(double val1, double val2) {
  EXPECT_PRED_FORMAT2(testing::DoubleLE, val1, val2);
}

template <
    typename Traits,
    int Dim,
    typename IndexType,
    typename SpaceType,
    typename SdimType,
    typename NptsType,
    typename ScalarType>
void CheckTraits(
    SpaceType const& space,
    SdimType sdim,
    NptsType npts,
    NptsType point_index,
    ScalarType const* point_data_ref) {
  static_assert(
      std::is_same<typename Traits::SpaceType, SpaceType>::value,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(Traits::Dim == Dim, "TRAITS_DIM_NOT_EQUAL_TO_EXPECTED_DIM");

  static_assert(
      std::is_same<typename Traits::IndexType, IndexType>::value,
      "TRAITS_INDEX_TYPE_INCORRECT");

  static_assert(
      std::is_same<typename Traits::ScalarType, ScalarType>::value,
      "TRAITS_SCALAR_TYPE_INCORRECT");

  EXPECT_EQ(static_cast<int>(sdim), Traits::SpaceSdim(space));
  EXPECT_EQ(static_cast<IndexType>(npts), Traits::SpaceNpts(space));

  ScalarType const* point_data_tst = Traits::PointCoords(
      Traits::PointAt(space, static_cast<IndexType>(point_index)));

  for (int i = 0; i < sdim; ++i) {
    FloatEq(point_data_ref[i], point_data_tst[i]);
  }
}

template <typename Traits, typename Index, typename Metric>
void SearchKnn(
    typename Traits::PointType const& point,
    typename Traits::SpaceType const& space,
    Index const k,
    Metric const& metric,
    std::vector<pico_tree::Neighbor<Index, typename Traits::ScalarType>>* knn) {
  //
  Index const npts = Traits::SpaceNpts(space);
  knn->resize(static_cast<std::size_t>(npts));
  for (Index i = 0; i < npts; ++i) {
    (*knn)[i] = {i, metric(point, Traits::PointAt(space, i))};
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
  using TraitsX = typename Tree::TraitsType;
  using SpaceX = typename Tree::SpaceType;
  using PointX = typename pico_tree::StdTraits<SpaceX>::PointType;
  using Index = typename Tree::IndexType;

  auto const points = tree.points();

  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;
  tree.SearchBox(min, max, &idxs);

  for (auto j : idxs) {
    for (int d = 0; d < PointX::Dim; ++d) {
      auto v = TraitsX::PointCoords(TraitsX::PointAt(points, j))[d];
      EXPECT_GE(v, min_v);
      EXPECT_LE(v, max_v);
    }
  }

  std::size_t count = 0;

  for (Index j = 0; j < TraitsX::SpaceNpts(points); ++j) {
    bool contained = true;

    for (int d = 0; d < PointX::Dim; ++d) {
      auto v = TraitsX::PointCoords(TraitsX::PointAt(points, j))[d];
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
  using TraitsX = typename Tree::TraitsType;
  using Index = typename Tree::IndexType;
  using Scalar = typename Tree::ScalarType;

  auto const points = tree.points();
  auto const p = TraitsX::PointAt(points, TraitsX::SpaceNpts(points) / 2);

  auto const& metric = tree.metric();
  Scalar const lp_radius = metric(radius);
  std::vector<pico_tree::Neighbor<Index, Scalar>> results;
  tree.SearchRadius(p, lp_radius, &results);

  for (auto const& r : results) {
    EXPECT_LE(metric(p, TraitsX::PointAt(points, r.index)), lp_radius);
    EXPECT_EQ(metric(p, TraitsX::PointAt(points, r.index)), r.distance);
  }

  std::size_t count = 0;

  for (Index j = 0; j < TraitsX::SpaceNpts(points); ++j) {
    if (metric(p, TraitsX::PointAt(points, j)) <= lp_radius) {
      count++;
    }
  }

  EXPECT_EQ(count, results.size());
}

template <typename Tree>
void TestKnn(Tree const& tree, typename Tree::IndexType const k) {
  using TraitsX = typename Tree::TraitsType;
  using Index = typename Tree::IndexType;
  using Scalar = typename Tree::ScalarType;

  // The data doesn't have to be by reference_wrapper, but that prevents a copy.
  auto const points = tree.points();
  auto const p = TraitsX::PointAt(points, TraitsX::SpaceNpts(points) / 2);
  Scalar ratio = tree.metric()(Scalar(1.5));

  std::vector<pico_tree::Neighbor<Index, Scalar>> results_exact;
  std::vector<pico_tree::Neighbor<Index, Scalar>> results_apprx;
  tree.SearchKnn(p, k, &results_exact);
  tree.SearchAknn(p, k, ratio, &results_apprx);

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
