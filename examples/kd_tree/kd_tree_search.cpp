#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Different search options.
void SearchR3() {
  using PointX = Point3f;
  using Scalar = typename PointX::ScalarType;

  std::size_t run_count = 1024 * 1024;
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  Scalar area_size = 1000;

  using KdTree = pico_tree::KdTree<std::vector<PointX>>;

  KdTree tree(GenerateRandomN<PointX>(point_count, area_size), max_leaf_size);

  Scalar min_v = 25.1f;
  Scalar max_v = 37.9f;
  PointX min, max, q;
  min.Fill(min_v);
  max.Fill(max_v);
  q.Fill((max_v + min_v) / 2.0f);

  Scalar search_radius = 2.0f;
  // When building KdTree with default template arguments, this is the squared
  // distance.
  Scalar search_radius_metric = tree.metric()(search_radius);
  // The ann can not be further away than a factor of (1 + max_error_percentage)
  // from the real nn.
  Scalar max_error_percentage = 0.2f;
  // Apply the metric to the max ratio difference.
  Scalar max_error_ratio_metric = tree.metric()(1.0f + max_error_percentage);

  using Neighbor = typename KdTree::NeighborType;
  using Index = typename Neighbor::IndexType;

  Neighbor nn;
  std::vector<Neighbor> knn;
  std::vector<Index> idxs;

  {
    ScopedTimer t("kd_tree nn, radius and box", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.SearchNn(q, nn);
      tree.SearchKnn(q, 1, knn);
      tree.SearchRadius(q, search_radius_metric, knn, false);
      tree.SearchBox(min, max, idxs);
    }
  }

  std::size_t k = 8;

  {
    ScopedTimer t("kd_tree aknn", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      // When the KdTree is created with the SlidingMidpointSplitter,
      // approximate nn queries can be answered in O(1/e^d log n) time.
      tree.SearchKnn(q, k, max_error_ratio_metric, knn);
    }
  }

  {
    ScopedTimer t("kd_tree knn", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.SearchKnn(q, k, knn);
    }
  }
}

// Search on the circle.
void SearchS1() {
  using PointX = Point1f;
  using SpaceX = std::vector<PointX>;
  using NeighborX = pico_tree::KdTree<SpaceX, pico_tree::SO2>::NeighborType;

  const auto pi = typename PointX::ScalarType(3.1415926537);

  pico_tree::KdTree<SpaceX, pico_tree::SO2> tree(
      GenerateRandomN<PointX>(512, -pi, pi), pico_tree::max_leaf_size_t(10));

  std::array<NeighborX, 8> knn;
  tree.SearchKnn(PointX{pi}, knn.begin(), knn.end());

  // These prints show that wrapping around near point -PI ~ PI is supported.
  std::cout << "Closest angles (index, distance, value): " << std::endl;
  for (auto const& nn : knn) {
    std::cout << "  " << nn.index << ", " << nn.distance << ", "
              << tree.points()[nn.index] << std::endl;
  }
}

int main() {
  SearchR3();
  SearchS1();
  return 0;
}
