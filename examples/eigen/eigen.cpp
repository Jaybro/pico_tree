#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/eigen3_traits.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/vector_traits.hpp>

// Because we use C++17 there is no need to take care of memory alignment:
// https://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
// https://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html

template <typename Point>
using PointsMapColMajor = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Point::RowsAtCompileTime,
    Eigen::Dynamic>>;

template <typename Point>
using PointsMapRowMajor = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Eigen::Dynamic,
    Point::ColsAtCompileTime,
    Eigen::RowMajor>>;

std::size_t const kRunCount = 1024 * 1024;
std::size_t const kNumPoints = 1024 * 1024 * 2;
float const kArea = 1000.0;
pico_tree::max_leaf_size_t const kMaxLeafCount = 16;

template <typename PointX>
std::vector<PointX> GenerateRandomEigenN(
    std::size_t n, typename PointX::Scalar size) {
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
  using Index = int;

  // Including <pico_tree/eigen3_traits.hpp> provides support for Eigen types
  // with std::vector.
  pico_tree::KdTree tree(
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
  using KdTree = pico_tree::KdTree<Eigen::Matrix3Xf>;
  using Neighbor = typename KdTree::NeighborType;
  using Scalar = typename Eigen::Matrix3Xf::Scalar;
  constexpr int Dim = Eigen::Matrix3Xf::RowsAtCompileTime;

  KdTree tree(
      Eigen::Matrix3Xf::Random(Dim, kNumPoints) * kArea / Scalar(2.0),
      kMaxLeafCount);

  Eigen::Vector3f p = Eigen::Vector3f::Random() * kArea / Scalar(2.0);
  Neighbor nn;
  ScopedTimer t("pico_tree eigen matrix", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, nn);
  }
}

// Creates a KdTree from a col-major matrix. The matrix maps an
// std::vector<Eigen::Vector3f>.
void ColMajorSupport() {
  using PointX = Eigen::Vector3f;
  using Map = PointsMapColMajor<PointX>;
  using KdTree = pico_tree::KdTree<Map>;
  using Neighbor = typename KdTree::NeighborType;
  using Scalar = typename PointX::Scalar;
  constexpr int Dim = PointX::RowsAtCompileTime;

  auto points = GenerateRandomEigenN<PointX>(kNumPoints, kArea);
  PointX p = PointX::Random() * kArea / Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Map::IsRowMajor << std::endl;

  KdTree tree(Map(points.data()->data(), Dim, points.size()), kMaxLeafCount);

  std::vector<Neighbor> knn;
  ScopedTimer t("pico_tree col major", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchKnn(p, 1, knn);
  }
}

// Creates a KdTree from a row-major matrix. The matrix maps an
// std::vector<Eigen::RowVector3f>.
void RowMajorSupport() {
  using PointX = Eigen::RowVector3f;
  using Map = PointsMapRowMajor<PointX>;
  using KdTree = pico_tree::KdTree<Map>;
  using Neighbor = typename KdTree::NeighborType;
  using Scalar = typename PointX::Scalar;
  constexpr int Dim = PointX::ColsAtCompileTime;

  auto points = GenerateRandomEigenN<PointX>(kNumPoints, kArea);
  PointX p = PointX::Random() * kArea / Scalar(2.0);

  std::cout << "Eigen RowMajor: " << PointX::IsRowMajor << std::endl;

  KdTree tree(Map(points.data()->data(), points.size(), Dim), kMaxLeafCount);

  std::vector<Neighbor> knn;
  ScopedTimer t("pico_tree row major", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchKnn(p, 1, knn);
  }
}

int main() {
  BasicVector();
  BasicMatrix();
  ColMajorSupport();
  RowMajorSupport();
  return 0;
}
