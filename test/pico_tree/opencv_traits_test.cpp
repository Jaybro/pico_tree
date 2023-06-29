#include <gtest/gtest.h>

#include <pico_tree/kd_tree.hpp>
#include <pico_tree/opencv_traits.hpp>

#include "common.hpp"

TEST(OpenCvTest, Interface) {
  using Scalar = float;
  int constexpr Dim = 3;

  cv::Mat matrix(8, 3, cv::DataType<Scalar>::type);
  cv::randu(matrix, -Scalar(1.0), Scalar(1.0));
  cv::Mat row = matrix.row(matrix.rows - 1);

  CheckSpaceAdaptor<Dim>(
      pico_tree::MatWrapper<Scalar, Dim>(matrix),
      matrix.cols,
      matrix.rows,
      matrix.rows - 1,
      row.ptr<Scalar>());
}

TEST(OpenCvTest, TreeCompatibility) {
  using Scalar = float;

  cv::Mat random(1024, 4, cv::DataType<Scalar>::type);
  cv::randu(random, -Scalar(1.0), Scalar(1.0));

  pico_tree::KdTree<pico_tree::MatWrapper<Scalar, 4>> tree(random, 10);

  TestKnn(tree, 20);
}
