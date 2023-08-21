#pragma once

#include <filesystem>
#include <locale>
#include <pico_tree/internal/stream.hpp>
#include <type_traits>

// http://corpus-texmex.irisa.fr/

namespace pico_tree {

namespace internal {

template <typename T>
inline std::string FormatString() {
  if constexpr (
      std::is_same_v<T, unsigned char> || std::is_same_v<T, std::byte>) {
    return "bvecs";
  } else if constexpr (std::is_same_v<T, float>) {
    return "fvecs";
  } else if constexpr (std::is_same_v<T, int>) {
    return "ivecs";
  } else {
    throw std::runtime_error(
        "Type shoule be one of unsigned char, float, or int.");
  }
}

inline std::string ToLower(std::string s) {
  std::transform(
      s.begin(), s.end(), s.begin(), [](auto c) { return std::tolower(c); });
  return s;
}

}  // namespace internal

template <typename T, std::size_t N>
void ReadXvecs(std::string const& filename, std::vector<std::array<T, N>>& v) {
  auto format_string = internal::FormatString<T>();
  auto filename_lower = internal::ToLower(filename);

  if (filename_lower.compare(
          filename_lower.size() - format_string.size(),
          format_string.size(),
          format_string) != 0) {
    throw std::runtime_error(
        "Filename expected to end with ." + format_string + ".");
  }

  std::fstream fstream =
      internal::OpenStream(filename, std::ios::in | std::ios::binary);
  internal::Stream stream(fstream);

  auto bytes = std::filesystem::file_size(filename);
  std::size_t const element_size = sizeof(T);
  std::size_t const row_size = sizeof(int) + element_size * N;
  std::size_t const row_count = static_cast<std::size_t>(bytes) / row_size;
  v.resize(row_count);
  for (auto& r : v) {
    int coords;
    stream.Read(coords);
    stream.Read(r);
  }
}

}  // namespace pico_tree
