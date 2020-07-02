#include <pico_point_set.hpp>
#include <pico_tree/kd_tree.hpp>
#include <scoped_timer.hpp>

#include "nano_point_set.hpp"

// *****************************************************************************
// Not a benchmark, just a simple comparison.
// *****************************************************************************

// Interestingly, at least for MinGW GCC 9.2.0, it has a positive effect on the
// average build time to be build first.
void BuildTree() {
  using PointXd = Point2d;
  using NanoPointSetXd = NanoPointSet<Index, PointXd>;
  using PicoPointSetXd = PicoPointSet<Index, PointXd>;

  int run_count = 4096;
  int max_leaf_count = 1;
  std::vector<PointXd> random = GenerateRandomN<PointXd>(1024, 100);
  NanoPointSetXd nano_points(random);
  PicoPointSetXd pico_points(random);

  {
    ScopedTimer t("tree gen nanoflann ct", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      NanoflannKdTree<NanoPointSetXd> kt(
          PointXd::Dims,
          nano_points,
          nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_count));
      kt.buildIndex();
    }
  }

  {
    ScopedTimer t("tree gen nanoflann rt", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      NanoflannKdTreeRt<NanoPointSetXd> kt(
          PointXd::Dims,
          nano_points,
          nanoflann::KDTreeSingleIndexAdaptorParams(max_leaf_count));
      kt.buildIndex();
    }
  }

  {
    ScopedTimer t("tree gen pico_tree ct", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      KdTree<PicoPointSetXd> rt(pico_points, max_leaf_count);
    }
  }

  {
    ScopedTimer t("tree gen pico_tree rt", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      KdTreeRt<PicoPointSetXd> rt(pico_points, max_leaf_count);
    }
  }
}

void SearchKnn() {
  using PointXd = Point3d;
  using NanoPointSetXd = NanoPointSet<Index, PointXd>;
  using PicoPointSetXd = PicoPointSet<Index, PointXd>;

  int run_count = 1024 * 1024;
  int point_count = 1024 * 1024;
  float area_size = 1000;

  int max_leaf_count = 8;
  std::vector<PointXd> random =
      GenerateRandomN<PointXd>(point_count, area_size);
  NanoPointSetXd nano_points(random);
  PicoPointSetXd pico_points(random);

  constexpr int k = 10;

  // nano flann
  Index indices[k];
  Scalar distances[k];
  std::vector<std::pair<Index, Scalar>> nf;
  nf.reserve(k);

  auto p = GenerateRandomP<PointXd>(area_size);
  Scalar v = 7.5f;
  Scalar radius = v * v;

  {
    NanoflannKdTree<NanoPointSetXd> kt(
        PointXd::Dims,
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
    KdTree<PicoPointSetXd> rt(pico_points, max_leaf_count);

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
