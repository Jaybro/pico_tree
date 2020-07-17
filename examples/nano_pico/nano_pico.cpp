#include <pico_point_set.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

#include "nano_point_set.hpp"

// *****************************************************************************
// Not a benchmark, just a simple comparison.
// *****************************************************************************

using Index = int;

// Interestingly, at least for MinGW GCC 9.2.0, it has a positive effect on the
// average build time to be build first.
void BuildTree() {
  using PointX = Point2f;
  using NanoPointSetX = NanoPointSet<Index, PointX>;
  using PicoPointSetX = PicoPointSet<Index, PointX>;

  int run_count = 4096;
  int max_leaf_count = 1;
  std::vector<PointX> random = GenerateRandomN<PointX>(1024, 100);
  NanoPointSetX nano_points(random);
  PicoPointSetX pico_points(random);

  {
    ScopedTimer t("tree gen nanoflann ct", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      NanoflannKdTree<NanoPointSetX> kt(
          PointX::Dims,
          nano_points,
          nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_count));
      kt.buildIndex();
    }
  }

  {
    ScopedTimer t("tree gen nanoflann rt", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      NanoflannKdTreeRt<NanoPointSetX> kt(
          PointX::Dims,
          nano_points,
          nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_count));
      kt.buildIndex();
    }
  }

  {
    ScopedTimer t("tree gen pico_tree ct", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      KdTree<PicoPointSetX> rt(pico_points, max_leaf_count);
    }
  }

  {
    ScopedTimer t("tree gen pico_tree rt", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      KdTreeRt<PicoPointSetX> rt(pico_points, max_leaf_count);
    }
  }
}

void SearchKnn() {
  using PointX = Point3f;
  using Scalar = PointX::Scalar;
  using NanoPointSetX = NanoPointSet<Index, PointX>;
  using PicoPointSetX = PicoPointSet<Index, PointX>;

  int run_count = 1024 * 1024;
  int point_count = 1024 * 1024;
  float area_size = 1000;

  int max_leaf_count = 8;
  std::vector<PointX> random = GenerateRandomN<PointX>(point_count, area_size);
  NanoPointSetX nano_points(random);
  PicoPointSetX pico_points(random);

  constexpr int k = 10;

  // nano flann
  Index indices[k];
  Scalar distances[k];
  std::vector<std::pair<Index, Scalar>> nf;
  nf.reserve(k);

  auto p = GenerateRandomP<PointX>(area_size);
  Scalar v = 7.5f;
  Scalar radius = v * v;

  {
    NanoflannKdTree<NanoPointSetX> kt(
        PointX::Dims,
        nano_points,
        nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_count));
    kt.buildIndex();

    ScopedTimer t("tree nn_ nanoflann", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      kt.knnSearch(p.data, 1, indices, distances);
      kt.knnSearch(p.data, k, indices, distances);
      kt.radiusSearch(p.data, radius, nf, nanoflann::SearchParams{0, 0, false});
    }
  }

  // pico tree
  std::pair<Index, Scalar> nn;
  std::vector<std::pair<Index, Scalar>> n;
  n.reserve(k);

  {
    KdTree<PicoPointSetX> rt(pico_points, max_leaf_count);

    ScopedTimer t("tree nn_ pico_tree", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      nn = rt.SearchNn(p);
      rt.SearchKnn(p, k, &n);
      rt.SearchRadius(p, radius, &n);
    }
  }
}

int main() {
  BuildTree();
  SearchKnn();
  return 0;
}
