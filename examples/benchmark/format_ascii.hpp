#pragma once

#include <iomanip>
#include <pico_tree/core.hpp>
#include <point.hpp>

namespace pico_tree {

template <typename T>
void WriteAscii(std::string const& filename, std::vector<T> const& v) {
  if (v.empty()) {
    return;
  }

  std::fstream stream = internal::OpenStream(filename, std::ios::out);
  stream << std::setprecision(10);
  for (auto const& p : v) {
    stream << p << '\n';
  }
}

}  // namespace pico_tree
