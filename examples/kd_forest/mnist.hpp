#pragma once

#include <algorithm>
#include <filesystem>
#include <pico_toolshed/format/format_mnist.hpp>

template <typename U_, typename T_, std::size_t N_>
std::array<U_, N_> cast(std::array<T_, N_> const& i) {
  std::array<U_, N_> c;
  std::transform(i.begin(), i.end(), c.begin(), [](T_ a) -> U_ {
    return static_cast<U_>(a);
  });
  return c;
}

template <typename U_, typename T_, std::size_t N_>
std::vector<std::array<U_, N_>> cast(std::vector<std::array<T_, N_>> const& i) {
  std::vector<std::array<U_, N_>> c;
  std::transform(
      i.begin(),
      i.end(),
      std::back_inserter(c),
      [](std::array<T_, N_> const& a) -> std::array<U_, N_> {
        return cast<U_>(a);
      });
  return c;
}

class mnist {
 private:
  using scalar_type = float;
  using image_byte = std::array<std::byte, 28 * 28>;
  using image_float = std::array<scalar_type, 28 * 28>;

  static std::vector<image_float> read_images(std::string const& filename) {
    if (!std::filesystem::exists(filename)) {
      throw std::runtime_error(filename + " doesn't exist");
    }

    std::vector<image_byte> images_u8;
    pico_tree::read_mnist_images(filename, images_u8);
    return cast<scalar_type>(images_u8);
  }

 public:
  using point_type = image_float;

  static std::string const dataset_name;

  static std::vector<point_type> read_train() {
    std::string fn_images_train = "train-images.idx3-ubyte";
    return read_images(fn_images_train);
  }

  static std::vector<point_type> read_test() {
    std::string fn_images_test = "t10k-images.idx3-ubyte";
    return read_images(fn_images_test);
  }
};

std::string const mnist::dataset_name = "mnist";
