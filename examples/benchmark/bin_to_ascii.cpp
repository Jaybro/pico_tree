#include <filesystem>
#include <pico_toolshed/format/format_ascii.hpp>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/point.hpp>

int main() {
  std::filesystem::path dirname_root = ".";
  std::filesystem::path filename_bin = "scans.bin";
  std::filesystem::path filename_ascii = "scans.txt";
  std::filesystem::path path_bin = dirname_root / filename_bin;
  std::filesystem::path path_ascii = dirname_root / filename_ascii;

  if (!std::filesystem::exists(path_bin)) {
    std::cout << path_bin.string() << " doesn't exist." << std::endl;
  } else if (!std::filesystem::exists(path_ascii)) {
    std::cout << "Reading points in bin format..." << std::endl;
    std::vector<pico_tree::point_3f> points;
    pico_tree::read_bin(path_bin.string(), points);
    std::cout << "Read " << points.size() << " points." << std::endl;
    std::cout << "Writing points to ascii xyz format..." << std::endl;
    pico_tree::write_ascii(path_ascii.string(), points);
    std::cout << "Done." << std::endl;
  } else {
    std::cout << path_ascii.string() << " already exists." << std::endl;
  }

  return 0;
}
