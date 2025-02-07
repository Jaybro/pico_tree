#pragma once

#include <filesystem>
#include <pico_tree/internal/stream_wrapper.hpp>

namespace pico_tree {

template <typename T_>
void write_bin(std::string const& filename, std::vector<T_> const& v) {
  if (v.empty()) {
    return;
  }

  std::size_t const element_size = sizeof(T_);
  std::fstream stream =
      internal::open_stream(filename, std::ios::out | std::ios::binary);
  stream.write(reinterpret_cast<char const*>(&v[0]), element_size * v.size());
}

template <typename T_>
void read_bin(std::string const& filename, std::vector<T_>& v) {
  std::fstream stream =
      internal::open_stream(filename, std::ios::in | std::ios::binary);

  auto bytes = std::filesystem::file_size(filename);
  std::size_t const element_size = sizeof(T_);
  std::size_t const element_count =
      static_cast<std::size_t>(bytes) / element_size;
  v.resize(element_count);
  stream.read(reinterpret_cast<char*>(&v[0]), element_size * element_count);
}

}  // namespace pico_tree
