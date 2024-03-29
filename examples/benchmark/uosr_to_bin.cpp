#include <filesystem>
#include <pico_toolshed/format/format_bin.hpp>
#include <pico_toolshed/format/format_uosr.hpp>

int main() {
  std::filesystem::path dirname_root = ".";
  std::filesystem::path filename_bin = "scans.bin";
  std::filesystem::path path_bin = dirname_root / filename_bin;

  if (!std::filesystem::exists(path_bin)) {
    std::cout << "Reading scans in uosr format..." << std::endl;
    std::vector<Point3f> points;
    pico_tree::ReadUosr(dirname_root.string(), points);
    std::cout << "Read " << points.size() << " points." << std::endl;
    std::cout << "Writing scans to bin xyz format..." << std::endl;
    pico_tree::WriteBin(path_bin.string(), points);
    std::cout << "Done." << std::endl;
  } else {
    std::cout << path_bin.string() << " already exists." << std::endl;
  }

  return 0;
}
