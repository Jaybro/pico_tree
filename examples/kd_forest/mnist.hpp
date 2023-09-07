#pragma once

#include <algorithm>
#include <filesystem>
#include <pico_toolshed/format/format_mnist.hpp>

template <typename U, typename T, std::size_t N>
std::array<U, N> Cast(std::array<T, N> const& i) {
  std::array<U, N> c;
  std::transform(i.begin(), i.end(), c.begin(), [](T a) -> U {
    return static_cast<U>(a);
  });
  return c;
}

template <typename U, typename T, std::size_t N>
std::vector<std::array<U, N>> Cast(std::vector<std::array<T, N>> const& i) {
  std::vector<std::array<U, N>> c;
  std::transform(
      i.begin(),
      i.end(),
      std::back_inserter(c),
      [](std::array<T, N> const& a) -> std::array<U, N> { return Cast<U>(a); });
  return c;
}

class Mnist {
 private:
  using Scalar = float;
  using ImageByte = std::array<std::byte, 28 * 28>;
  using ImageFloat = std::array<Scalar, 28 * 28>;

  static std::vector<ImageFloat> ReadImages(std::string const& filename) {
    if (!std::filesystem::exists(filename)) {
      throw std::runtime_error(filename + " doesn't exist.");
    }

    std::vector<ImageByte> images_u8;
    pico_tree::ReadMnistImages(filename, images_u8);
    return Cast<Scalar>(images_u8);
  }

 public:
  using PointType = ImageFloat;

  static std::string const kDatasetName;

  static std::vector<PointType> ReadTrain() {
    std::string fn_images_train = "train-images.idx3-ubyte";
    return ReadImages(fn_images_train);
  }

  static std::vector<PointType> ReadTest() {
    std::string fn_images_test = "t10k-images.idx3-ubyte";
    return ReadImages(fn_images_test);
  }
};

std::string const Mnist::kDatasetName = "mnist";
