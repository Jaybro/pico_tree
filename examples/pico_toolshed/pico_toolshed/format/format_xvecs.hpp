#pragma once

#include <filesystem>
#include <locale>
#include <pico_tree/internal/stream_wrapper.hpp>
#include <type_traits>

// http://corpus-texmex.irisa.fr/

namespace pico_tree {

namespace internal {

template <typename T_>
inline std::string format_string() {
  if constexpr (
      std::is_same_v<T_, unsigned char> || std::is_same_v<T_, std::byte>) {
    return "bvecs";
  } else if constexpr (std::is_same_v<T_, float>) {
    return "fvecs";
  } else if constexpr (std::is_same_v<T_, int>) {
    return "ivecs";
  } else {
    throw std::runtime_error(
        "Type shoule be one of unsigned char, float, or int.");
  }
}

inline std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](auto c) {
    return std::tolower(c, std::locale());
  });
  return s;
}

}  // namespace internal

template <typename T_, std::size_t N_>
void read_xvecs(
    std::string const& filename, std::vector<std::array<T_, N_>>& v) {
  auto format_string = internal::format_string<T_>();
  auto filename_lower = internal::to_lower(filename);

  if (filename_lower.compare(
          filename_lower.size() - format_string.size(),
          format_string.size(),
          format_string) != 0) {
    throw std::runtime_error(
        "Filename expected to end with ." + format_string + ".");
  }

  std::fstream fstream =
      internal::open_stream(filename, std::ios::in | std::ios::binary);
  internal::stream_wrapper stream(fstream);

  auto bytes = std::filesystem::file_size(filename);
  std::size_t const element_size = sizeof(T_);
  std::size_t const row_size = sizeof(int) + element_size * N_;
  std::size_t const row_count = static_cast<std::size_t>(bytes) / row_size;
  v.resize(row_count);
  for (auto& r : v) {
    int coords;
    stream.read(coords);
    stream.read(r);
  }
}

}  // namespace pico_tree
