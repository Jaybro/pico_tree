#include <Eigen/Dense>
#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>
#include <random>
#include <scoped_timer.hpp>

using Index = int;
using Scalar = double;

std::size_t const kRunCount = 1024 * 1024;
int const kNumPoints = 1024 * 1024;
double const kArea = 1000.0;
Index const kMaxLeafCount = 16;

template <typename Point>
std::vector<Point, Eigen::aligned_allocator<Point>> GenerateRandomEigenN(
    int n, typename Point::Scalar size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, size);

  std::vector<Point, Eigen::aligned_allocator<Point>> random(n);
  for (auto& p : random) {
    p = Point::Random() * size / typename Point::Scalar(2.0);
  }

  return random;
}

void ColMajor() {
  using Point = Eigen::Vector3d;
  constexpr int Dim = Point::RowsAtCompileTime;
  using PointsMap =
      Eigen::Map<Eigen::Matrix<Point::Scalar, Dim, Eigen::Dynamic>>;
  using Adaptor = pico_tree::EigenAdaptor<Index, PointsMap>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap points_map(points.data()->data(), Dim, points.size());
  Adaptor adaptor(points_map);

  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> rt(
        adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      rt.SearchKnn(p, 1, &knn);
    }
  }
}

void RowMajor() {
  using Point = Eigen::RowVector3d;
  constexpr int Dim = Point::ColsAtCompileTime;
  using PointsMap = Eigen::Map<
      Eigen::Matrix<Point::Scalar, Eigen::Dynamic, Dim, Eigen::RowMajor>>;
  using Adaptor = pico_tree::EigenAdaptor<Index, PointsMap>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap points_map(points.data()->data(), points.size(), Dim);
  Adaptor adaptor(points_map);

  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> rt(
        adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      rt.SearchKnn(p, 1, &knn);
    }
  }
}

void Metrics() {
  using Point = Eigen::Vector3d;
  constexpr int Dim = Point::RowsAtCompileTime;
  using PointsMap =
      Eigen::Map<Eigen::Matrix<Point::Scalar, Dim, Eigen::Dynamic>>;
  using Adaptor = pico_tree::EigenAdaptor<Index, PointsMap>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap points_map(points.data()->data(), Dim, points.size());
  Adaptor adaptor(points_map);

  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen Metrics: " << std::endl;

  {
    pico_tree::KdTree<
        Index,
        Point::Scalar,
        Dim,
        Adaptor,
        pico_tree::EigenMetricL2<Point::Scalar>>
        rt(adaptor, kMaxLeafCount);

    {
      pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> rt(
          adaptor, kMaxLeafCount);

      std::vector<std::pair<Index, Scalar>> knn;
      ScopedTimer t("tree nn_ pico_tree eigen l2", kRunCount);
      for (std::size_t i = 0; i < kRunCount; ++i) {
        rt.SearchKnn(p, 1, &knn);
      }
    }
  }

  {
    pico_tree::KdTree<
        Index,
        Point::Scalar,
        Dim,
        Adaptor,
        pico_tree::EigenMetricL1<Point::Scalar>>
        rt(adaptor, kMaxLeafCount);

    {
      pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> rt(
          adaptor, kMaxLeafCount);

      std::vector<std::pair<Index, Scalar>> knn;
      ScopedTimer t("tree nn_ pico_tree eigen l1", kRunCount);
      for (std::size_t i = 0; i < kRunCount; ++i) {
        rt.SearchKnn(p, 1, &knn);
      }
    }
  }
}

int main() {
  RowMajor();
  ColMajor();
  Metrics();
  return 0;
}
