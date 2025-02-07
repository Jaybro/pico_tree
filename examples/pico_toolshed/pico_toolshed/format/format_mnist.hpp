#pragma once

#include <array>
#include <cstdint>
#include <pico_tree/internal/stream_wrapper.hpp>

#include "endian.hpp"

// http://yann.lecun.com/exdb/mnist/

namespace pico_tree {

namespace internal {

struct mnist_images {
  struct header {
    static constexpr std::int32_t magic_number = 2051;
    std::int32_t image_count;
    std::int32_t image_width;
    std::int32_t image_height;
  };
};

struct mnist_labels {
  struct header {
    static constexpr std::int32_t magic_number = 2049;
    std::int32_t label_count;
  };
};

}  // namespace internal

inline void read_mnist_images(
    std::string const& filename,
    std::vector<std::array<std::byte, 28 * 28>>& images) {
  using namespace internal;

  std::fstream stream = open_stream(filename, std::ios::in | std::ios::binary);
  stream_wrapper wrapper(stream);

  big_endian<std::int32_t> magic_number;
  wrapper.read(magic_number);
  if (magic_number() != mnist_images::header::magic_number) {
    throw std::runtime_error(
        "Incorrect signature: Expected MNIST images magic number.");
  }

  mnist_images::header header;
  wrapper.read(header);
  header.image_count = big_endian<std::int32_t>{header.image_count};
  header.image_width = big_endian<std::int32_t>{header.image_width};
  header.image_height = big_endian<std::int32_t>{header.image_height};

  if (header.image_width * header.image_height != 28 * 28) {
    throw std::runtime_error("Unexpected MNIST image dimensions.");
  }

  images.resize(header.image_count);
  wrapper.read(images.size(), images.data());
}

inline void read_mnist_labels(
    std::string const& filename, std::vector<std::byte>& labels) {
  using namespace internal;

  std::fstream stream = open_stream(filename, std::ios::in | std::ios::binary);
  stream_wrapper wrapper(stream);

  big_endian<std::int32_t> magic_number;
  wrapper.read(magic_number);
  if (magic_number() != mnist_labels::header::magic_number) {
    throw std::runtime_error(
        "Incorrect signature: Expected MNIST labels magic number.");
  }

  mnist_labels::header header;
  wrapper.read(header);
  header.label_count = big_endian<std::int32_t>{header.label_count};

  labels.resize(header.label_count);
  wrapper.read(labels.size(), labels.data());
}

}  // namespace pico_tree
