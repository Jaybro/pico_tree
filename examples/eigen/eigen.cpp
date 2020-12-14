#include <Eigen/Dense>
// This example compiles with C++11.
// Using C++11 and higher don't need the StdVector include (as mentioned inside
// the include itself).
//#include <Eigen/StdVector>
// If we use C++17 there is no need to take care of memory alignment:
// https://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>
#include <random>
#include <scoped_timer.hpp>

// Important! This is not a performance benchmark. So don't take the "elapsed
// time" numbers too seriously.

using Index = int;
// This fixed type size requires us to use aligned memory.
// https://eigen.tuxfamily.org/dox-devel/group__TopicFixedSizeVectorizable.html
using PointCm = Eigen::Vector4f;
using PointRm = Eigen::RowVector4f;
using Scalar = typename PointCm::Scalar;

template <typename Point>
using PointsMapCm = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Point::RowsAtCompileTime,
    Eigen::Dynamic>>;  //, Eigen::AlignedMax>;
// The alignment used by Eigen equals Eigen::AlignedMax. Note that Eigen can
// look at the data pointer to know if it is properly aligned.

template <typename Point>
using PointsMapRm = Eigen::Map<Eigen::Matrix<
    typename Point::Scalar,
    Eigen::Dynamic,
    Point::ColsAtCompileTime,
    Eigen::RowMajor>>;  //, Eigen::AlignedMax>;

template <typename Point>
using EigenAdaptorCm = pico_tree::EigenAdaptor<Index, PointsMapCm<Point>>;

template <typename Point>
using EigenAdaptorRm = pico_tree::EigenAdaptor<Index, PointsMapRm<Point>>;

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
  using Point = PointCm;
  constexpr int Dim = Point::RowsAtCompileTime;
  using PointsMap = PointsMapCm<Point>;
  using Adaptor = EigenAdaptorCm<Point>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> tree(
        Adaptor(PointsMap(points.data()->data(), Dim, points.size())),
        kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }
}

void RowMajor() {
  using Point = PointRm;
  constexpr int Dim = Point::ColsAtCompileTime;
  using PointsMap = PointsMapRm<Point>;
  using Adaptor = EigenAdaptorRm<Point>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen RowMajor: " << Adaptor::RowMajor << std::endl;

  {
    pico_tree::KdTree<Index, Point::Scalar, Dim, Adaptor> tree(
        Adaptor(PointsMap(points.data()->data(), points.size(), Dim)),
        kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree deflt l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }
}

void Metrics() {
  using Point = PointCm;
  constexpr int Dim = Point::RowsAtCompileTime;
  using PointsMap = PointsMapCm<Point>;
  using Adaptor = EigenAdaptorCm<Point>;

  auto points = GenerateRandomEigenN<Point>(kNumPoints, kArea);
  PointsMap map(points.data()->data(), Dim, points.size());
  Adaptor adaptor(map);
  Point p = Point::Random() * kArea / typename Point::Scalar(2.0);

  std::cout << "Eigen Metrics: " << std::endl;

  {
    pico_tree::KdTree<
        Index,
        Point::Scalar,
        Dim,
        Adaptor,
        pico_tree::EigenMetricL2<Scalar>>
        tree(adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree eigen l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }

  {
    pico_tree::KdTree<
        Index,
        Point::Scalar,
        Dim,
        Adaptor,
        pico_tree::EigenMetricL1<Scalar>>
        tree(adaptor, kMaxLeafCount);

    std::vector<std::pair<Index, Scalar>> knn;
    ScopedTimer t("tree nn_ pico_tree eigen l1", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }
}

int main() {
  RowMajor();
  ColMajor();
  Metrics();
  return 0;
}
