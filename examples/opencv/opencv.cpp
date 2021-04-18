#include <pico_toolshed/scoped_timer.hpp>
#include <pico_tree/kd_tree.hpp>
#include <pico_tree/opencv.hpp>
#include <random>

using Index = int;
using Scalar = float;

Index const kNumPoints = 1024 * 1024 * 2;
Scalar const kArea = 1000.0;
std::size_t const kRunCount = 1024 * 1024;

template <typename Scalar_>
std::vector<cv::Point3_<Scalar_>> GenerateRandomPoint3N(int n, Scalar_ size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<Scalar_> dist(0, size);

  std::vector<cv::Point3_<Scalar_>> random(n);
  for (auto& p : random) {
    p.x = dist(e2);
    p.y = dist(e2);
    p.z = dist(e2);
  }

  return random;
}

// This example shows to build a KdTree from a vector of cv::Point3.
void BasicVector() {
  using PointX = cv::Point3_<Scalar>;
  std::vector<PointX> random = GenerateRandomPoint3N(kNumPoints, kArea);

  pico_tree::KdTree<
      pico_tree::StdTraits<std::reference_wrapper<std::vector<PointX>>>>
      tree(random, 10);

  auto p = random[random.size() / 2];

  pico_tree::Neighbor<Index, Scalar> nn;
  ScopedTimer t("pico_tree cv vector", kRunCount);
  for (std::size_t i = 0; i < kRunCount; ++i) {
    tree.SearchNn(p, &nn);
  }
}

// This example shows to build a KdTree using a cv::Mat.
void BasicMatrix() {
  // Multiple columns based on the amount of coordinates in a point.
  {
    cv::Mat random(kNumPoints, 3, CV_32FC1);
    cv::randu(random, Scalar(0.0), kArea);

    pico_tree::KdTree<pico_tree::CvTraits<Scalar, 3>> tree(random, 10);

    pico_tree::CvMatRow<Scalar, 3> p(tree.points().rows / 2, tree.points());

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree cv mat", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, &nn);
    }
  }

  // Single column cv::Mat based on a vector of points.
  {
    using PointX = cv::Point3_<Scalar>;
    std::vector<PointX> random = GenerateRandomPoint3N(kNumPoints, kArea);

    pico_tree::KdTree<pico_tree::CvTraits<Scalar, 3>> tree(cv::Mat(random), 10);

    PointX p = random[random.size() / 2];

    pico_tree::Neighbor<Index, Scalar> nn;
    ScopedTimer t("pico_tree cv mat", kRunCount);
    for (std::size_t i = 0; i < kRunCount; ++i) {
      tree.SearchNn(p, &nn);
    }
  }
}

int main() {
  BasicVector();
  BasicMatrix();
  return 0;
}
