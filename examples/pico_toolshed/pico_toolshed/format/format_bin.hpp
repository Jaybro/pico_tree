#pragma once

#include <filesystem>
#include <pico_tree/internal/stream_wrapper.hpp>

namespace pico_tree {

template <typename T_>
void write_bin(std::string const& filename, std::vector<T_> const& v) {
  if (v.empty()) {
    return;
  }

  std::fstream stream =
      internal::open_stream(filename, std::ios::out | std::ios::binary);
  internal::stream_wrapper wrapper(stream);
  wrapper.write(v.data(), v.size());
}

template <typename T_>
void read_bin(std::string const& filename, std::vector<T_>& v) {
  std::fstream stream =
      internal::open_stream(filename, std::ios::in | std::ios::binary);
  internal::stream_wrapper wrapper(stream);

  auto bytes = std::filesystem::file_size(filename);
  std::size_t const element_size = sizeof(T_);
  std::size_t const element_count =
      static_cast<std::size_t>(bytes) / element_size;
  v.resize(element_count);
  wrapper.read(element_count, v.data());
}

}  // namespace pico_tree
