#include <Eigen/Dense>
#include <pico_tree/eigen_adaptor.hpp>
#include <pico_tree/kd_tree.hpp>
#include <point.hpp>
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

template <typename Adaptor>
void AdaptorCout(Adaptor const& a, Index idx) {
  Index num_dims = a.num_dimensions();
  for (Index i = 0; i < num_dims; ++i) std::cout << a(idx, i) << " ";
  std::cout << std::endl;
}

void ColMajor() {
  using Point = Eigen::Vector3d;
  constexpr int Dims = Point::RowsAtCompileTime;
  using PointsMap =
      Eigen::Map<Eigen::Matrix<Point::Scalar, Dims, Eigen::Dynamic>>;
  using Adaptor = pico_tree::EigenAdaptor<Index, PointsMap>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap points_map(points.data()->data(), Dims, points.size());
  Adaptor adaptor(points_map);

  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << points[points.size() - 1].transpose() << std::endl;
  AdaptorCout(adaptor, points.size() - 1);
  std::cout << "RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dims, Adaptor> rt(
        adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      rt.SearchKnn(p, 1, &knn);
    }
  }
}

void RowMajor() {
  using Point = Eigen::RowVector3d;
  constexpr int Dims = Point::ColsAtCompileTime;
  using PointsMap = Eigen::Map<
      Eigen::Matrix<Point::Scalar, Eigen::Dynamic, Dims, Eigen::RowMajor>>;
  using Adaptor = pico_tree::EigenAdaptor<Index, PointsMap>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap points_map(points.data()->data(), points.size(), Dims);
  Adaptor adaptor(points_map);

  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << points[points.size() - 1] << std::endl;
  AdaptorCout(adaptor, points.size() - 1);
  std::cout << "RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dims, Adaptor> rt(
        adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      rt.SearchKnn(p, 1, &knn);
    }
  }
}

int main() {
  RowMajor();
  ColMajor();
  return 0;
}
