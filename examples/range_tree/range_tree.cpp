#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/range_tree.hpp>
#include <scoped_timer.hpp>

template <typename PicoAdaptor>
using KdTree = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using RangeTree2d = pico_tree::RangeTree2d<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor>;

int main() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index run_count = 1024 * 1024;
  Index max_leaf_size = 1;
  Index point_count = 1024 * 1024;
  Scalar const area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoAdaptorX adaptor(random);

  Scalar const min_v = 25.1f;
  Scalar const max_v = 37.9f;
  PointX min, max;
  min.Fill(min_v);
  max.Fill(max_v);

  std::vector<Index> idxs;

  {
    KdTree<PicoAdaptorX> tree(adaptor, max_leaf_size);

    ScopedTimer t("tree rq kd_tree", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchBox(min, max, &idxs);
    }
  }

  {
    RangeTree2d<PicoAdaptorX> tree(adaptor);

    ScopedTimer t("tree rq rg_tree", run_count);
    for (Index i = 0; i < run_count; ++i) {
      tree.SearchBox(min, max, &idxs);
      // TODO The KdTree clears the list inside SearchBox. The RangeTree
      // doesn't do this yet.
      idxs.clear();
    }
  }

  return 0;
}
