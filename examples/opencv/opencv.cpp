#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/opencv_traits.hpp>
#include <pico_tree/vector_traits.hpp>
#include <random>

using index = int;
using scalar = float;

index const num_points = 1024 * 1024 * 2;
scalar const point_area = 1000.0;
std::size_t const run_count = 1024 * 1024;

template <typename Vec_>
std::vector<Vec_> generate_random_vec_n(
    std::size_t n, typename Vec_::value_type size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<typename Vec_::value_type> dist(0, size);

  std::vector<Vec_> random(n);
  for (auto& p : random) {
    for (auto& c : p.val) {
      c = dist(e2);
    }
  }

  return random;
}

// This example shows to build a kd_tree from a vector of cv::Point3.
void basic_vector() {
  using point = cv::Vec<scalar, 3>;
  std::vector<point> random =
      generate_random_vec_n<point>(num_points, point_area);

  pico_tree::kd_tree tree(std::cref(random), pico_tree::max_leaf_size_t(10));

  auto p = random[random.size() / 2];

  pico_tree::neighbor<index, scalar> nn;
  pico_tree::scoped_timer t("pico_tree cv vector", run_count);
  for (std::size_t i = 0; i < run_count; ++i) {
    tree.search_nn(p, nn);
  }
}

// This example shows to build a kd_tree using a cv::Mat.
void basic_matrix() {
  // Multiple columns based on the number of coordinates in a point.
  {
    constexpr int dim = 3;
    cv::Mat random(num_points, dim, cv::DataType<scalar>::type);
    cv::randu(random, scalar(0.0), point_area);

    pico_tree::kd_tree<pico_tree::opencv_mat_map<scalar, dim>> tree(
        random, pico_tree::max_leaf_size_t(10));
    pico_tree::point_map<scalar, dim> p = tree.points()[random.rows / 2];

    pico_tree::neighbor<index, scalar> nn;
    pico_tree::scoped_timer t("pico_tree cv mat", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.search_nn(p, nn);
    }
  }

  // Single column cv::Mat based on a vector of points.
  {
    using point = cv::Vec<scalar, 3>;
    std::vector<point> random =
        generate_random_vec_n<point>(num_points, point_area);

    pico_tree::kd_tree<pico_tree::opencv_mat_map<scalar, 3>> tree(
        cv::Mat(random), pico_tree::max_leaf_size_t(10));

    point p = random[random.size() / 2];

    pico_tree::neighbor<index, scalar> nn;
    pico_tree::scoped_timer t("pico_tree cv mat", run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
      tree.search_nn(p, nn);
    }
  }
}

int main() {
  basic_vector();
  basic_matrix();
  return 0;
}
