#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/eigen3_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Because we use C++17 there is no need to take care of memory alignment:
// https://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
// https://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html

template <typename Point_>
using points_map_col_major = Eigen::Map<Eigen::Matrix<
    typename Point_::Scalar,
    Point_::RowsAtCompileTime,
    Eigen::Dynamic>>;

template <typename Point_>
using points_map_row_major = Eigen::Map<Eigen::Matrix<
    typename Point_::Scalar,
    Eigen::Dynamic,
    Point_::ColsAtCompileTime,
    Eigen::RowMajor>>;

std::size_t const run_count = 1024 * 1024;
std::size_t const num_points = 1024 * 1024 * 2;
float const point_area = 1000.0;
pico_tree::max_leaf_size_t const max_leaf_count = 16;

template <typename Point_>
std::vector<Point_> generate_random_eigen_n(
    std::size_t n, typename Point_::Scalar size) {
  std::vector<Point_> random(n);
  for (auto& p : random) {
    p = Point_::Random() * size / typename Point_::Scalar(2.0);
  }

  return random;
}

// Creates a kd_tree from a vector of Eigen::VectorX and searches for nearest
// neighbors.
void basic_vector() {
  using point = Eigen::Vector2f;
  using scalar = typename point::Scalar;
  using index = int;

  // Including <pico_tree/eigen3_traits.hpp> provides support for Eigen types
  // with std::vector.
  pico_tree::kd_tree tree(
      generate_random_eigen_n<point>(num_points, point_area), max_leaf_count);

  point p = point::Random() * point_area / scalar(2.0);

  pico_tree::neighbor<index, scalar> nn;
  pico_tree::scoped_timer t("pico_tree eigen vector", run_count);
  for (std::size_t i = 0; i < run_count; ++i) {
    tree.search_nn(p, nn);
  }
}

// Creates a kd_tree from an Eigen::Matrix<> and searches for nearest neighbors.
void basic_matrix() {
  using kd_tree = pico_tree::kd_tree<Eigen::Matrix3Xf>;
  using neighbor = typename kd_tree::neighbor_type;
  using scalar = typename Eigen::Matrix3Xf::Scalar;
  constexpr int dim = Eigen::Matrix3Xf::RowsAtCompileTime;

  kd_tree tree(
      Eigen::Matrix3Xf::Random(dim, num_points) * point_area / scalar(2.0),
      max_leaf_count);

  Eigen::Vector3f p = Eigen::Vector3f::Random() * point_area / scalar(2.0);
  neighbor nn;
  pico_tree::scoped_timer t("pico_tree eigen matrix", run_count);
  for (std::size_t i = 0; i < run_count; ++i) {
    tree.search_nn(p, nn);
  }
}

// Creates a kd_tree from a col-major matrix. The matrix maps an
// std::vector<Eigen::Vector3f>.
void col_major_support() {
  using point = Eigen::Vector3f;
  using map = points_map_col_major<point>;
  using kd_tree = pico_tree::kd_tree<map>;
  using neighbor = typename kd_tree::neighbor_type;
  using scalar = typename point::Scalar;
  constexpr int dim = point::RowsAtCompileTime;

  auto points = generate_random_eigen_n<point>(num_points, point_area);
  point p = point::Random() * point_area / scalar(2.0);

  std::cout << "Eigen RowMajor: " << map::IsRowMajor << std::endl;

  kd_tree tree(map(points.data()->data(), dim, points.size()), max_leaf_count);

  std::vector<neighbor> knn;
  pico_tree::scoped_timer t("pico_tree col major", run_count);
  for (std::size_t i = 0; i < run_count; ++i) {
    tree.search_knn(p, 1, knn);
  }
}

// Creates a kd_tree from a row-major matrix. The matrix maps an
// std::vector<Eigen::RowVector3f>.
void row_major_support() {
  using point = Eigen::RowVector3f;
  using map = points_map_row_major<point>;
  using kd_tree = pico_tree::kd_tree<map>;
  using neighbor = typename kd_tree::neighbor_type;
  using scalar = typename point::Scalar;
  constexpr int dim = point::ColsAtCompileTime;

  auto points = generate_random_eigen_n<point>(num_points, point_area);
  point p = point::Random() * point_area / scalar(2.0);

  std::cout << "Eigen RowMajor: " << point::IsRowMajor << std::endl;

  kd_tree tree(map(points.data()->data(), points.size(), dim), max_leaf_count);

  std::vector<neighbor> knn;
  pico_tree::scoped_timer t("pico_tree row major", run_count);
  for (std::size_t i = 0; i < run_count; ++i) {
    tree.search_knn(p, 1, knn);
  }
}

int main() {
  basic_vector();
  basic_matrix();
  col_major_support();
  row_major_support();
  return 0;
}
