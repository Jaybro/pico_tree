#include <gtest/gtest.h>

#include <pico_tree/opencv_traits.hpp>

#include "common.hpp"

TEST(OpenCvTraitsTest, Interface) {
  using scalar_type = float;
  constexpr pico_tree::size_t dim = 3;

  cv::Mat matrix(8, 3, cv::DataType<scalar_type>::type);
  cv::randu(matrix, -scalar_type(1.0), scalar_type(1.0));
  cv::Mat row = matrix.row(matrix.rows - 1);
  pico_tree::opencv_mat_map<scalar_type, dim> space(matrix);

  check_space_adaptor<dim>(
      space,
      static_cast<pico_tree::size_t>(matrix.cols),
      static_cast<pico_tree::size_t>(matrix.rows),
      static_cast<pico_tree::size_t>(matrix.rows - 1),
      row.ptr<scalar_type>());

  static_assert(space.sdim() == dim);
}
