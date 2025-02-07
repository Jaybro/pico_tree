#include <filesystem>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/format/format_uosr.hpp>

int main() {
  std::filesystem::path dirname_root = ".";
  std::filesystem::path filename_bin = "scans.bin";
  std::filesystem::path path_bin = dirname_root / filename_bin;

  if (!std::filesystem::exists(path_bin)) {
    std::cout << "Reading scans in uosr format..." << std::endl;
    std::vector<pico_tree::point_3f> points;
    pico_tree::read_uosr(dirname_root.string(), points);
    std::cout << "Read " << points.size() << " points." << std::endl;
    std::cout << "Writing scans to bin xyz format..." << std::endl;
    pico_tree::write_bin(path_bin.string(), points);
    std::cout << "Done." << std::endl;
  } else {
    std::cout << path_bin.string() << " already exists." << std::endl;
  }

  return 0;
}
