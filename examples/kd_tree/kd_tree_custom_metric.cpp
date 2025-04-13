#include <pico_toolshed/point.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// This example shows how to create a custom metric for the kd_tree. The kd_tree
// does not support metrics that "normalize" the distance between two points for
// performance reasons. This means that, for example, the L2 metric is not
// supported, but that there is support for the L1 or L2^2 metric.
//
// There are two different categories of metrics: Euclidean and topological. A
// Euclidean metric supports any R^n space where all axes are orthogonal with
// respect to each other. A topological space is the same, but it allows
// wrapping of coordinate values along any of the axes. An implementation of a
// custom metric is provided for both below.

// The LP^P metric is a generalization of the L2^2 metric.
template <std::size_t P_>
struct metric_lp_p {
  static_assert(P_ > 0, "P_CANNOT_BE_ZERO");

  // Indicate that this metric is a Euclidean one.
  using space_category = pico_tree::euclidean_space_tag;

  // Calculates the distance between two points.
  template <typename InputIterator_>
  auto operator()(
      InputIterator_ begin1, InputIterator_ end1, InputIterator_ begin2) const {
    using scalar_type =
        typename std::iterator_traits<InputIterator_>::value_type;

    scalar_type d{};
    for (; begin1 != end1; ++begin1, ++begin2) {
      d += operator()(*begin1 - *begin2);
    }
    return d;
  }

  // Returns the absolute value of x to the power of p.
  template <typename Scalar_>
  Scalar_ operator()(Scalar_ x) const {
    return std::pow(std::abs(x), static_cast<Scalar_>(P_));
  }
};

// This metric measures distances on the two dimensional ring torus T2. The
// torus is the Cartesian product of two circles S1 x S1. The values of each of
// the point coordinates should be within the range of [0...1].
struct metric_t2_squared {
  // Indicate that this metric is defined on a topological space. The
  // topological_space_tag is required because the torus wraps around in both
  // dimensions.
  using space_category = pico_tree::topological_space_tag;

  // Calculates the distance between two points.
  template <typename InputIterator_>
  auto operator()(
      InputIterator_ begin1, InputIterator_ end1, InputIterator_ begin2) const {
    using scalar_type =
        typename std::iterator_traits<InputIterator_>::value_type;

    scalar_type d{};
    for (; begin1 != end1; ++begin1, ++begin2) {
      d += pico_tree::squared_s1_distance(*begin1, *begin2);
    }
    return d;
  }

  // Distances are squared values.
  template <typename Scalar_>
  Scalar_ operator()(Scalar_ x) const {
    return x * x;
  }

  template <typename UnaryPredicate_>
  void apply_dim_space([[maybe_unused]] int dim, UnaryPredicate_ p) const {
    p(pico_tree::one_space_s1{});
  }
};

void search_lp3_3() {
  using point = pico_tree::point_2f;
  using scalar = typename point::scalar_type;

  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  scalar area_size = 10;

  using kd_tree = pico_tree::kd_tree<std::vector<point>, metric_lp_p<3>>;
  using neighbor = typename kd_tree::neighbor_type;

  kd_tree tree(
      pico_tree::generate_random_n<point>(point_count, area_size),
      max_leaf_size);

  neighbor nn;
  tree.search_nn(point{area_size / scalar(2), area_size / scalar(2)}, nn);

  std::cout << "Index closest point: " << nn.index << std::endl;
}

void search_t2() {
  using point = pico_tree::point_2f;
  using scalar = typename point::scalar_type;

  pico_tree::max_leaf_size_t max_leaf_size = 12;
  std::size_t point_count = 1024 * 1024;
  scalar area_size = 1;

  using kd_tree = pico_tree::kd_tree<std::vector<point>, metric_t2_squared>;
  using neighbor = typename kd_tree::neighbor_type;

  kd_tree tree(
      pico_tree::generate_random_n<point>(point_count, area_size),
      max_leaf_size);

  std::array<neighbor, 8> knn;
  tree.search_knn(point{area_size, area_size}, knn.begin(), knn.end());

  // These prints show that wrapping near values 0 ~ 1 is supported.
  std::cout << "Closest points (index, distance, point): " << std::endl;
  for (auto const& nn : knn) {
    std::cout << "  " << nn.index << ", " << nn.distance << ", ["
              << tree.space()[static_cast<std::size_t>(nn.index)] << "]"
              << std::endl;
  }
}

int main() {
  search_lp3_3();
  search_t2();
  return 0;
}
