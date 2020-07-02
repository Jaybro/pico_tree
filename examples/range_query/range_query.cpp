#include <pico_point_set.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/range_tree.hpp>
#include <scoped_timer.hpp>

int main() {
  using PointXd = Point2d;
  using PicoPointSetXd = PicoPointSet<Index, PointXd>;

  int run_count = 1024 * 1024;
  Index max_leaf_count = 1;
  Index point_count = 1024 * 1024;
  Scalar const area_size = 1000;
  std::vector<PointXd> random =
      GenerateRandomN<PointXd>(point_count, area_size);
  PicoPointSetXd points(random);

  Scalar const min_v = 15.1f;
  Scalar const max_v = 34.9f;
  PointXd min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;

  {
    KdTree<PicoPointSetXd> rt(points, max_leaf_count);

    ScopedTimer t("tree rq kd_tree", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      rt.SearchRange(min, max, &idxs);
    }
  }

  {
    RangeTree2d<PicoPointSetXd> rt(points);

    ScopedTimer t("tree rq rg_tree", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      rt.SearchRange(min, max, &idxs);
    }
  }

  return 0;
}