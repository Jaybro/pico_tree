#include "open_stream.hpp"

#include <iostream>

namespace pico_tree {

std::fstream OpenStream(
    std::string const& filename, std::ios_base::openmode mode) {
  std::fstream stream(filename, mode);

  if (!stream.is_open()) {
    std::cout << "Unable to open file: " << filename << '\n';
    std::exit(EXIT_FAILURE);
  }

  return stream;
}

}  // namespace pico_tree
