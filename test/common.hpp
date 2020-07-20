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
  tree.SearchRange(min, max, &idxs);

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
