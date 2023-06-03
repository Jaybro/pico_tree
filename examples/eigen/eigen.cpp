#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/eigen3_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/std_traits.hpp>

// NOTES:

// Because we use C++17 there is no need to take care of memory alignment:
// https://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
// https://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html

// The Eigen example is not a performance benchmark. So don't take the "elapsed
// time" numbers too seriously.

using Index = int;

template <typename Point>
using PointsMapCm = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Point::RowsAtCompileTime,
    Eigen::Dynamic>>;

template <typename Point>
using PointsMapRm = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Eigen::Dynamic,
    Point::ColsAtCompileTime,
    Eigen::RowMajor>>;

std::size_t const kRunCount = 1024 * 1024;
int const kNumPoints = 1024 * 1024 * 2;
float const kArea = 1000.0;
Index const kMaxLeafCount = 16;

template <typename PointX>
std::vector<PointX> GenerateRandomEigenN(int n, typename PointX::Scalar size) {
  std::vector<PointX> random(n);
  for (auto& p : random) {
    p = PointX::Random() * size / typename PointX::Scalar(2.0);
  }

  return random;
}

// Creates a KdTree from a vector of Eigen::VectorX and searches for nearest
// neighbors.
void BasicVector() {
  using PointX = Eigen::Vector2f;
  using Scalar = typename PointX::Scalar;

  // Including <pico_tree/eigen.hpp> provides support for Eigen types with
  // std::vector.
  pico_tree::KdTree<pico_tree::StdTraits<std::vector<PointX>>> tree(
      GenerateRandomEigenN<PointX>(kNumPoints, kArea), kMaxLeafCount);

  PointX p = PointX::Random() * kArea / Scalar(2.0);

  pico_tree::Neighbor<Index, Scalar> nn;
  ScopedTimer t("pico_tree eigen vector", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, nn);
  }
}

// Creates a KdTree from an Eigen::Matrix<> and searches for nearest neighbors.
void BasicMatrix() {
  using Scalar = typename Eigen::Matrix3Xf::Scalar;
  constexpr int Dim = Eigen::Matrix3Xf::RowsAtCompileTime;

  Eigen::Vector3f p = Eigen::Vector3f::Random() * kArea / Scalar(2.0);

  // The KdTree takes the matrix by value. Prevent a copy by:
  // * Using a move.
  // * Creating an Eigen::Map<>.
  // * Wrap with an std::reference_wrapper<>.
  {
    pico_tree::KdTree<pico_tree::EigenTraits<Eigen::Matrix3Xf>> tree(
        Eigen::Matrix3Xf::Random(Dim, kNumPoints) * kArea / Scalar(2.0),
        kMaxLeafCount);

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree eigen val", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, nn);
    }
  }

  {
    Eigen::Matrix3Xf matrix =
        Eigen::Matrix3Xf::Random(Dim, kNumPoints) * kArea / Scalar(2.0);

    pico_tree::KdTree<
        pico_tree::EigenTraits<std::reference_wrapper<Eigen::Matrix3Xf>>>
        tree(matrix, kMaxLeafCount);

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree eigen ref", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, nn);
    }
  }
}

// Creates a KdTree from a col-major matrix. The matrix maps an
// std::vector<Eigen::Vector3f>.
void VectorMapColMajor() {
  using PointX = Eigen::Vector3f;
  using Scalar = typename PointX::Scalar;
  constexpr int Dim = PointX::RowsAtCompileTime;
  using Map = PointsMapCm<PointX>;

  auto points = GenerateRandomEigenN<PointX>(kNumPoints, kArea);
  PointX p = PointX::Random() * kArea / Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Map::IsRowMajor << std::endl;
  {
    pico_tree::KdTree<pico_tree::EigenTraits<Map>> tree(
        Map(points.data()->data(), Dim, points.size()), kMaxLeafCount);

    std::vector<pico_tree::Neighbor<Index, Scalar>> knn;
    ScopedTimer t("pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, knn);
    }
  }
}

// Creates a KdTree from a row-major matrix. The matrix maps an
// std::vector<Eigen::RowVector3f>.
void VectorMapRowMajor() {
  using PointX = Eigen::RowVector3f;
  using Scalar = typename PointX::Scalar;
  constexpr int Dim = PointX::ColsAtCompileTime;
  using Map = PointsMapRm<PointX>;

  auto points = GenerateRandomEigenN<PointX>(kNumPoints, kArea);
  PointX p = PointX::Random() * kArea / Scalar(2.0);

  std::cout << "Eigen RowMajor: " << PointX::IsRowMajor << std::endl;

  {
    pico_tree::KdTree<pico_tree::EigenTraits<Map>> tree(
        Map(points.data()->data(), points.size(), Dim), kMaxLeafCount);

    std::vector<pico_tree::Neighbor<Index, Scalar>> knn;
    ScopedTimer t("pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, knn);
    }
  }
}

int main() {
  BasicVector();
  BasicMatrix();
  VectorMapColMajor();
  VectorMapRowMajor();
  return 0;
}
