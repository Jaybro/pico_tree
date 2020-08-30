#include <pico_point_set.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/range_tree.hpp>
#include <scoped_timer.hpp>

int main() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoPointSetX = PicoPointSet<Index, PointX>;

  int run_count = 1024 * 1024;
  Index max_leaf_count = 1;
  Index point_count = 1024 * 1024;
  Scalar const area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoPointSetX points(random);

  Scalar const min_v = 25.1f;
  Scalar const max_v = 37.9f;
  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;

  {
    KdTree<PicoPointSetX> tree(points, max_leaf_count);

    ScopedTimer t("tree rq kd_tree", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchRange(min, max, &idxs);
    }
  }

  {
    RangeTree2d<PicoPointSetX> tree(points);

    ScopedTimer t("tree rq rg_tree", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchRange(min, max, &idxs);
      // TODO The KdTree clears the list inside SearchRange. The RangeTree
      // doesn't do this yet.
      idxs.clear();
    }
  }

  return 0;
}