#pragma once

#include <array>
#include <cstdint>
#include <pico_tree/internal/stream.hpp>

#include "endian.hpp"

// http://yann.lecun.com/exdb/mnist/

namespace pico_tree {

namespace internal {

struct MnistImages {
  struct Header {
    static std::int32_t constexpr kMagicNumber = 2051;
    std::int32_t image_count;
    std::int32_t image_width;
    std::int32_t image_height;
  };
};

struct MnistLabels {
  struct Header {
    static std::int32_t constexpr kMagicNumber = 2049;
    std::int32_t label_count;
  };
};

}  // namespace internal

inline void ReadMnistImages(
    std::string const& filename,
    std::vector<std::array<std::byte, 28 * 28>>& images) {
  using namespace internal;

  std::fstream stream = OpenStream(filename, std::ios::in | std::ios::binary);
  Stream wrapper(stream);

  BigEndian<std::int32_t> magic_number;
  wrapper.Read(magic_number);
  if (magic_number() != MnistImages::Header::kMagicNumber) {
    throw std::runtime_error(
        "Incorrect signature: Expected MNIST images magic number.");
  }

  MnistImages::Header header;
  wrapper.Read(header);
  header.image_count = BigEndian<std::int32_t>{header.image_count};
  header.image_width = BigEndian<std::int32_t>{header.image_width};
  header.image_height = BigEndian<std::int32_t>{header.image_height};

  if (header.image_width * header.image_height != 28 * 28) {
    throw std::runtime_error("Unexpected MNIST image dimensions.");
  }

  images.resize(header.image_count);
  wrapper.Read(images.size(), images.data());
}

inline void ReadMnistLabels(
    std::string const& filename, std::vector<std::byte>& labels) {
  using namespace internal;

  std::fstream stream = OpenStream(filename, std::ios::in | std::ios::binary);
  Stream wrapper(stream);

  BigEndian<std::int32_t> magic_number;
  wrapper.Read(magic_number);
  if (magic_number() != MnistLabels::Header::kMagicNumber) {
    throw std::runtime_error(
        "Incorrect signature: Expected MNIST labels magic number.");
  }

  MnistLabels::Header header;
  wrapper.Read(header);
  header.label_count = BigEndian<std::int32_t>{header.label_count};

  labels.resize(header.label_count);
  wrapper.Read(labels.size(), labels.data());
}

}  // namespace pico_tree
