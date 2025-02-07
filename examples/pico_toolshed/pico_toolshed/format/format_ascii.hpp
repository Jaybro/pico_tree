#pragma once

#include <iomanip>
#include <pico_tree/internal/stream_wrapper.hpp>

namespace pico_tree {

template <typename T_>
void write_ascii(std::string const& filename, std::vector<T_> const& v) {
  if (v.empty()) {
    return;
  }

  std::fstream stream = internal::open_stream(filename, std::ios::out);
  stream << std::setprecision(10);
  for (auto const& p : v) {
    stream << p << '\n';
  }
}

}  // namespace pico_tree
