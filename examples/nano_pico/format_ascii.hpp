#pragma once

#include <iomanip>
#include <point.hpp>

#include "open_stream.hpp"

namespace pico_tree {

template <typename T>
void WriteAscii(std::string const& filename, std::vector<T> const& v) {
  if (v.empty()) {
    return;
  }

  std::fstream stream = OpenStream(filename, std::ios::out);
  stream << std::setprecision(10);
  for (auto const& p : v) {
    stream << p << '\n';
  }
}

}  // namespace pico_tree
