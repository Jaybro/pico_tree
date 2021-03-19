#pragma once

#include <limits>
#include <pico_tree/core.hpp>

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
void ReadBin(std::string const& filename, std::vector<T>* v) {
  std::fstream stream =
      internal::OpenStream(filename, std::ios::in | std::ios::binary);

  // The four lines below are used when determining the file size.
  // C++17 is not used here in order to keep the benchmark C++11.
  stream.ignore(std::numeric_limits<std::streamsize>::max());
  std::streamsize byte_count = stream.gcount();
  stream.clear();
  stream.seekg(0, std::ios_base::beg);

  if (byte_count == 0) {
    return;
  }

  std::size_t const element_size = sizeof(T);
  std::size_t const element_count =
      static_cast<std::size_t>(byte_count / element_size);
  v->resize(element_count);
  stream.read(reinterpret_cast<char*>(&(*v)[0]), element_size * element_count);
}

}  // namespace pico_tree
