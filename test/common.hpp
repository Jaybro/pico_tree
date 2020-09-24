#pragma once

// TODO Perhaps do something more friendly for exposing types.
// TODO Some traits.
template <typename Tree>
void TestRange(
    Tree const& tree,
    typename std::remove_reference_t<
        decltype(std::declval<Tree>().points())>::Scalar const min_v,
    typename std::remove_reference_t<
        decltype(std::declval<Tree>().points())>::Scalar const max_v) {
  using PointsX =
      std::remove_reference_t<decltype(std::declval<Tree>().points())>;
  using PointX = typename PointsX::Point;
  using Index = typename PointsX::Index;

  auto const& points = tree.points();

  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;
  tree.SearchBox(min, max, &idxs);

  for (auto j : idxs) {
    for (int d = 0; d < PointX::Dims; ++d) {
      EXPECT_GE(points(j, d), min_v);
      EXPECT_LE(points(j, d), max_v);
    }
  }

  std::size_t count = 0;

  for (Index j = 0; j < points.num_points(); ++j) {
    bool contained = true;

    for (int d = 0; d < PointX::Dims; ++d) {
      if ((points(j, d) < min_v) || (points(j, d) > max_v)) {
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
void TestRadius(
    Tree const& tree,
    typename std::remove_reference_t<
        decltype(std::declval<Tree>().points())>::Scalar const radius) {
  using PointsX =
      std::remove_reference_t<decltype(std::declval<Tree>().points())>;
  using PointX = typename PointsX::Point;
  using Index = typename PointsX::Index;
  using Scalar = typename PointsX::Scalar;

  auto const& points = tree.points();

  Index idx = tree.points().num_points() / 2;
  PointX p;

  for (Index d = 0; d < PointsX::Dims; ++d) {
    p(d) = points(idx, d);
  }

  auto const& metric = tree.metric();
  Scalar const lp_radius = metric(radius);
  std::vector<std::pair<Index, Scalar>> results;
  tree.SearchRadius(p, lp_radius, &results);

  for (auto const& r : results) {
    EXPECT_LE(metric(p, r.first), lp_radius);
    EXPECT_EQ(metric(p, r.first), r.second);
  }

  std::size_t count = 0;

  for (Index j = 0; j < points.num_points(); ++j) {
    if (metric(p, j) <= lp_radius) {
      count++;
    }
  }

  EXPECT_EQ(count, results.size());
}
