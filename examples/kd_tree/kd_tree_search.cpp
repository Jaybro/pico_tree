#include <pico_toolshed/point.hpp>
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Different search options.
void search_r3() {
  using point = pico_tree::point_3f;
  using scalar = typename point::scalar_type;

  std::size_t run_count = 1024 * 1024;
  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  scalar area_size = 1000;

  using kd_tree = pico_tree::kd_tree<std::vector<point>>;

  kd_tree tree(
      pico_tree::generate_random_n<point>(point_count, area_size),
      max_leaf_size);

  scalar min_v = 25.1f;
  scalar max_v = 37.9f;
  point min, max, q;
  min.fill(min_v);
  max.fill(max_v);
  q.fill((max_v + min_v) / 2.0f);

  scalar search_radius = 2.0f;
  // When building kd_tree with default template arguments, this is the squared
  // distance.
  scalar search_radius_metric = tree.metric()(search_radius);
  // The ann can not be further away than a factor of (1 + max_error_percentage)
  // from the real nn.
  scalar max_error_percentage = 0.2f;
  // Apply the metric to the max ratio difference.
  scalar max_error_ratio_metric = tree.metric()(1.0f + max_error_percentage);

  using neighbor = typename kd_tree::neighbor_type;
  using index = typename neighbor::index_type;

  neighbor nn;
  std::vector<neighbor> knn;
  std::vector<index> idxs;

  {
    pico_tree::scoped_timer t("kd_tree nn, radius and box", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.search_nn(q, nn);
      tree.search_knn(q, 1, knn);
      tree.search_radius(q, search_radius_metric, knn, false);
      tree.search_box(min, max, idxs);
    }
  }

  std::size_t k = 8;

  {
    pico_tree::scoped_timer t("kd_tree aknn", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      // When the kd_tree is created with the sliding_midpoint_max_side splitter
      // rule (the default argument), approximate nn queries can be answered in
      // O(1/e^d log n) time.
      tree.search_knn(q, k, max_error_ratio_metric, knn);
    }
  }

  {
    pico_tree::scoped_timer t("kd_tree knn", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.search_knn(q, k, knn);
    }
  }
}

// This example shows how to search on the unit circle. Point coordinates must
// lie within the range of [0...1]. Point coordinates wrap at 0 or 1. A point
// with a coordinate value of 0 is considered the same as one with a coordinate
// value of 1.
void search_s1() {
  using point = pico_tree::point_1f;
  using scalar = typename pico_tree::point_1f::scalar_type;
  using space = std::vector<point>;
  using neighbor =
      pico_tree::kd_tree<space, pico_tree::metric_so2>::neighbor_type;

  pico_tree::kd_tree<space, pico_tree::metric_so2> tree(
      pico_tree::generate_random_n<point>(512, scalar(0.0), scalar(1.0)),
      pico_tree::max_leaf_size_t(10));

  std::array<neighbor, 8> knn;
  tree.search_knn(point{1.0}, knn.begin(), knn.end());

  // These prints show that wrapping near values 0 ~ 1 is supported.
  std::cout << "Closest angles (index, distance, value): " << std::endl;
  for (auto const& nn : knn) {
    std::cout << "  " << nn.index << ", " << nn.distance << ", "
              << tree.space()[static_cast<std::size_t>(nn.index)] << std::endl;
  }
}

int main() {
  search_r3();
  search_s1();
  return 0;
}
