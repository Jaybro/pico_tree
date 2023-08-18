#pragma once

#include <filesystem>
#include <pico_tree/internal/stream.hpp>

namespace pico_tree {

template <typename T>
void WriteBin(std::string const& filename, std::vector<T> const& v) {
  if (v.empty()) {
    return;
  }

  std::size_t const element_size = sizeof(T);
  std::fstream stream =
      internal::OpenStream(filename, std::ios::out | std::ios::binary);
  stream.write(reinterpret_cast<char const*>(&v[0]), element_size * v.size());
}

template <typename T>
void ReadBin(std::string const& filename, std::vector<T>& v) {
  std::fstream stream =
      internal::OpenStream(filename, std::ios::in | std::ios::binary);

  auto bytes = std::filesystem::file_size(filename);
  std::size_t const element_size = sizeof(T);
  std::size_t const element_count =
      static_cast<std::size_t>(bytes) / element_size;
  v.resize(element_count);
  stream.read(reinterpret_cast<char*>(&v[0]), element_size * element_count);
}

}  // namespace pico_tree
