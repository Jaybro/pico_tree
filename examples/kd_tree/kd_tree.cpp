#include <pico_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

template <typename PicoAdaptor>
using KdTreeCt = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    PicoAdaptor::Dims,
    PicoAdaptor>;

template <typename PicoAdaptor>
using KdTreeRt = pico_tree::KdTree<
    typename PicoAdaptor::Index,
    typename PicoAdaptor::Scalar,
    pico_tree::kRuntimeDims,
    PicoAdaptor>;

// Compile time or run time known dimensions.
void Build() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoAdaptorX adaptor(random);

  {
    ScopedTimer t("build kd_tree ct");
    KdTreeCt<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }

  {
    ScopedTimer t("build kd_tree rt");
    KdTreeRt<PicoAdaptorX> tree(adaptor, max_leaf_size);
  }
}

// Different search options.
void Searches() {
  using PointX = Point2f;
  using Index = int;
  using Scalar = typename PointX::Scalar;
  using PicoAdaptorX = PicoAdaptor<Index, PointX>;

  Index run_count = 1024 * 1024;
  Index max_leaf_size = 12;
  Index point_count = 1024 * 1024;
  Scalar area_size = 1000;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  PicoAdaptorX adaptor(random);
  KdTreeCt<PicoAdaptorX> tree(adaptor, max_leaf_size);

  Scalar min_v = 25.1f;
  Scalar max_v = 37.9f;
  PointX min, max, pnn;
  min.Fill(min_v);
  max.Fill(max_v);
  pnn.Fill((max_v + min_v) / 2.0f);

  Index k = 4;
  Scalar search_radius = 2.0;
  Scalar search_radius_metric = tree.metric()(search_radius);

  std::vector<std::pair<Index, Scalar>> nn;
  std::vector<Index> idxs;

  ScopedTimer t("kd_tree", run_count);
  for (Index i = 0; i < run_count; ++i) {
    tree.SearchKnn(pnn, k, &nn, false);
    tree.SearchRadius(pnn, search_radius_metric, &nn, false);
    tree.SearchRange(min, max, &idxs);
  }
}

int main() {
  Build();
  Searches();
  return 0;
}
