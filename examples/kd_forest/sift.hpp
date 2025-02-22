#pragma once

#include <filesystem>
#include <pico_toolshed/format/format_xvecs.hpp>

class sift {
 private:
  using vector_float = std::array<float, 128>;

  static std::vector<vector_float> read_vectors(std::string const& filename) {
    if (!std::filesystem::exists(filename)) {
      throw std::runtime_error(filename + " doesn't exist");
    }

    std::vector<vector_float> vectors;
    pico_tree::read_xvecs(filename, vectors);
    return vectors;
  }

 public:
  using point_type = vector_float;

  static std::string const dataset_name;

  static std::vector<point_type> read_train() {
    std::string fn_images_train = "sift_base.fvecs";
    return read_vectors(fn_images_train);
  }

  static std::vector<point_type> read_test() {
    std::string fn_images_test = "sift_query.fvecs";
    return read_vectors(fn_images_test);
  }
};

std::string const sift::dataset_name = "sift";
