// This example compiles with C++11.
// C++11 and higher don't need the StdVector include (as mentioned inside the
// include itself).
//#include <Eigen/StdVector>
// If we use C++17 there is no need to take care of memory alignment:
// https://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/eigen.hpp>
#include <pico_tree/kd_tree.hpp>

// Important! The Eigen example is not a performance benchmark. So don't take
// the "elapsed time" numbers too seriously.

using Index = int;

template <typename Point>
using PointsMapCm = Eigen::Map<
    Eigen::Matrix<
        typename Point::Scalar,
        Point::RowsAtCompileTime,
        Eigen::Dynamic>,
    Eigen::AlignedMax>;
// The alignment used by Eigen equals Eigen::AlignedMax. Note that Eigen can
// look at the data pointer to know if it is properly aligned.

template <typename Point>
using PointsMapRm = Eigen::Map<
    Eigen::Matrix<
        typename Point::Scalar,
        Eigen::Dynamic,
        Point::ColsAtCompileTime,
        Eigen::RowMajor>,
    Eigen::AlignedMax>;

std::size_t const kRunCount = 1024 * 1024;
int const kNumPoints = 1024 * 1024 * 2;
double const kArea = 1000.0;
Index const kMaxLeafCount = 16;

// Certain fixed size matrices require us to use aligned memory.
// https://eigen.tuxfamily.org/dox-devel/group__TopicFixedSizeVectorizable.html
template <typename PointX>
std::vector<PointX, Eigen::aligned_allocator<PointX>> GenerateRandomEigenN(
    int n, typename PointX::Scalar size) {
  std::vector<PointX, Eigen::aligned_allocator<PointX>> random(n);
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

  // Including <pico_tree/eigen.hpp> provices support for Eigen types with
  // std::vector.
  pico_tree::KdTree<pico_tree::StdTraits<
      std::vector<PointX, Eigen::aligned_allocator<PointX>>>>
      tree(GenerateRandomEigenN<PointX>(kNumPoints, kArea), kMaxLeafCount);

  PointX p = PointX::Random() * kArea / Scalar(2.0);

  pico_tree::Neighbor<Index, Scalar> nn;
  ScopedTimer t("pico_tree eigen vector", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, &nn);
  }
}

// Creates a KdTree from an Eigen::Matrix<> and searches for nearest neighbors.
void BasicMatrix() {
  using Scalar = typename Eigen::Matrix3Xf::Scalar;
  constexpr int Dim = Eigen::Matrix3Xf::RowsAtCompileTime;

  // The KdTree takes the matrix by value. Prevent a copy by:
  // * Using a move.
  // * Creating an Eigen::Map<>.
  // TODO Could add an std::reference_wrapper<> version for Eigen.
  pico_tree::KdTree<pico_tree::EigenTraits<Eigen::Matrix3Xf>> tree(
      Eigen::Matrix3Xf::Random(Dim, kNumPoints) * kArea / Scalar(2.0),
      kMaxLeafCount);

  Eigen::Vector3f p = Eigen::Vector3f::Random() * kArea / Scalar(2.0);

  pico_tree::Neighbor<Index, Scalar> nn;
  ScopedTimer t("pico_tree eigen matrix", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, &nn);
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
      tree.SearchKnn(p, 1, &knn);
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
      tree.SearchKnn(p, 1, &knn);
    }
  }
}

// The Metrics demo shows how it can be beneficial to use the metrics supplied
// by the <pico_tree/eigen.hpp> header.
//
// Suppose we want to use a KdTree with a spatial dimension of 3 using floats as
// a scalar. In this case we can use Eigen::Vector3f as the data type. However,
// Eigen::Vector3f doesn't benefit from vectorization. With Eigen::Vector4f we
// can, but in this case we have one dimension too many!
//
// Luckily, it is possible use a different dimension for both the points and the
// KdTree, but some care needs to be taken:
// * The default Metrics don't make explicit use of vectorization (perhaps
// implicitly through optimization by the compiler) but the Eigen based Metrics
// may do so.
// * The extra coordinate of Eigen::Vector4f must be set 0 so it doesn't
// influence any of the distance calculations. E.g., the squared distance uses a
// dot product.
//
// See also:
// http://eigen.tuxfamily.org/index.php?title=UsingVector4fForVector3fOperations
void Metrics() {
  // Eigen::Vector4f requires aligned memory.
  using PointX = Eigen::Vector4f;
  using Scalar = typename PointX::Scalar;
  using Map = PointsMapCm<PointX>;
  // Tell the KdTree to use a spatial dimension of 3 instead of 4.
  constexpr int Dim = PointX::RowsAtCompileTime - 1;

  auto points = GenerateRandomEigenN<PointX>(kNumPoints, kArea);
  // The Eigen::Map uses the dimension of 4.
  Map map(points.data()->data(), PointX::RowsAtCompileTime, points.size());
  // Set the last row (4th coordinate) to 0.
  map.bottomRows<1>().setZero();

  PointX p = PointX::Random() * kArea / Scalar(2.0);
  // Again, set the last row (4th coordinate) to 0.
  p.w() = Scalar(0.0);

  std::cout << "Eigen Metrics: " << std::endl;

  {
    // Using an std::reference_wrapper prevents a copy.
    using Traits = pico_tree::StdTraits<std::reference_wrapper<
        std::vector<PointX, Eigen::aligned_allocator<PointX>>>>;

    pico_tree::KdTree<
        Traits,
        pico_tree::EigenL2Squared<Scalar>,
        pico_tree::SplitterSlidingMidpoint<Traits>,
        Dim>
        tree(points, kMaxLeafCount);

    std::vector<pico_tree::Neighbor<Index, Scalar>> knn;
    ScopedTimer t("pico_tree eigen l2", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }

  {
    // Using an Eigen::Map prevents a copy.
    using Traits = pico_tree::EigenTraits<Map>;

    pico_tree::KdTree<
        Traits,
        pico_tree::EigenL1<Scalar>,
        pico_tree::SplitterSlidingMidpoint<Traits>,
        Dim>
        tree(map, kMaxLeafCount);

    std::vector<pico_tree::Neighbor<Index, Scalar>> knn;
    ScopedTimer t("pico_tree eigen l1", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchKnn(p, 1, &knn);
    }
  }
}

int main() {
  BasicVector();
  BasicMatrix();
  VectorMapColMajor();
  VectorMapRowMajor();
  Metrics();
  return 0;
}
