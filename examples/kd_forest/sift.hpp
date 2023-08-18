#pragma once

#include <filesystem>
#include <pico_toolshed/format/format_xvecs.hpp>

class Sift {
 private:
  using VectorFloat = std::array<float, 128>;

  static std::vector<VectorFloat> ReadVectors(std::string const& filename) {
    if (!std::filesystem::exists(filename)) {
      throw std::runtime_error(filename + " doesn't exist.");
    }

    std::vector<VectorFloat> vectors;
    pico_tree::ReadXvecs(filename, vectors);
    return vectors;
  }

 public:
  using PointType = VectorFloat;

  static std::string const kDatasetName;

  static std::vector<PointType> ReadTrain() {
    std::string fn_images_train = "sift_base.fvecs";
    return ReadVectors(fn_images_train);
  }

  static std::vector<PointType> ReadTest() {
    std::string fn_images_test = "sift_query.fvecs";
    return ReadVectors(fn_images_test);
  }
};

std::string const Sift::kDatasetName = "sift";
