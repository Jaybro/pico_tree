#include <gtest/gtest.h>

#include <pico_tree/kd_tree.hpp>
#include <pico_tree/opencv.hpp>

#include "common.hpp"

TEST(OpenCvTest, Interface) {
  using Scalar = float;
  int constexpr Dim = 3;
  using Traits = pico_tree::CvTraits<Scalar, Dim>;

  static_assert(
      std::is_same<typename Traits::SpaceType, cv::Mat>::value,
      "TRAITS_SPACE_TYPE_INCORRECT");

  static_assert(Traits::Dim == Dim, "TRAITS_DIM_NOT_EQUAL_TO_EXPECTED_DIM");

  static_assert(
      std::is_same<typename Traits::IndexType, int>::value,
      "TRAITS_INDEX_TYPE_NOT_INT");

  int rows = 8;
  int cols = 3;
  cv::Mat matrix(rows, cols, cv::DataType<Scalar>::type);

  EXPECT_EQ(cols, Traits::SpaceSdim(matrix));
  EXPECT_EQ(rows, Traits::SpaceNpts(matrix));
}

TEST(OpenCvTest, TreeCompatibility) {
  using Scalar = float;
  cv::Mat random(8, 4, cv::DataType<Scalar>::type);
  cv::randu(random, -Scalar(1.0), Scalar(1.0));

  pico_tree::KdTree<pico_tree::CvTraits<Scalar, 4>> tree(random, 10);

  TestKnn(tree, 2);
}
