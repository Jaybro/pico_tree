#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/opencv_traits.hpp>
#include <pico_tree/vector_traits.hpp>
#include <random>

using Index = int;
using Scalar = float;

Index const kNumPoints = 1024 * 1024 * 2;
Scalar const kArea = 1000.0;
std::size_t const kRunCount = 1024 * 1024;

template <typename Vec_>
std::vector<Vec_> GenerateRandomVecN(
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

// This example shows to build a KdTree from a vector of cv::Point3.
void BasicVector() {
  using PointX = cv::Vec<Scalar, 3>;
  std::vector<PointX> random = GenerateRandomVecN<PointX>(kNumPoints, kArea);

  pico_tree::KdTree tree(std::cref(random), 10);

  auto p = random[random.size() / 2];

  pico_tree::Neighbor<Index, Scalar> nn;
  ScopedTimer t("pico_tree cv vector", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, nn);
  }
}

// This example shows to build a KdTree using a cv::Mat.
void BasicMatrix() {
  // Multiple columns based on the number of coordinates in a point.
  {
    cv::Mat random(kNumPoints, 3, cv::DataType<Scalar>::type);
    cv::randu(random, Scalar(0.0), kArea);

    pico_tree::KdTree<pico_tree::MatWrapper<Scalar, 3>> tree(random, 10);
    pico_tree::PointMap<Scalar, 3> p(random.ptr<Scalar>(random.rows / 2));

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree cv mat", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, nn);
    }
  }

  // Single column cv::Mat based on a vector of points.
  {
    using PointX = cv::Vec<Scalar, 3>;
    std::vector<PointX> random = GenerateRandomVecN<PointX>(kNumPoints, kArea);

    pico_tree::KdTree<pico_tree::MatWrapper<Scalar, 3>> tree(
        cv::Mat(random), 10);

    PointX p = random[random.size() / 2];

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree cv mat", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, nn);
    }
  }
}

int main() {
  BasicVector();
  BasicMatrix();
  return 0;
}
