#include "format_bin.hpp"

#include <limits>

#include "open_stream.hpp"

namespace pico_tree {

void WriteBin(std::string const& filename, std::vector<Point3d> const& points) {
  if (points.empty()) {
    return;
  }

  std::size_t const point_size = sizeof(Point3d);
  std::fstream stream = OpenStream(filename, std::ios::out | std::ios::binary);
  stream.write(
      reinterpret_cast<const char*>(&points[0]), point_size * points.size());
}

std::vector<Point3d> ReadBin(std::string const& filename) {
  std::fstream stream = OpenStream(filename, std::ios::in | std::ios::binary);

  // The four lines below are used when determining the file size.
  // C++17 is not used here in order to keep nano_pico C++11.
  stream.ignore(std::numeric_limits<std::streamsize>::max());
  std::streamsize byte_count = stream.gcount();
  stream.clear();
  stream.seekg(0, std::ios_base::beg);

  if (byte_count == 0) {
    return {};
  }

  std::size_t const point_size = sizeof(Point3d);
  std::size_t const point_count =
      static_cast<std::size_t>(byte_count / point_size);
  std::vector<Point3d> points(point_count);
  stream.read(reinterpret_cast<char*>(&points[0]), point_size * point_count);

  return points;
}

}  // namespace pico_tree
