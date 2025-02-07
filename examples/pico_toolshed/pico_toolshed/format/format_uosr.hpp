#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/stream_wrapper.hpp>
#include <regex>
#include <string>

namespace pico_tree {

namespace internal {

constexpr auto string_scan = "scan";
constexpr auto string_pose = ".pose";
constexpr auto string_3d = ".3d";
constexpr double deg_to_rad = 3.1415926535897932384626433832795 / 180.0;

// Transformation that preserves distances between points. Column major.
class isometry {
 public:
  isometry(std::array<double, 9> const& R, std::array<double, 3> const& t)
      : R_{R}, t_{t} {}

  static isometry from_euler(
      std::array<double, 3> const& e, std::array<double, 3> const& t) {
    double sx = std::sin(e[0]);
    double cx = std::cos(e[0]);
    double sy = std::sin(e[1]);
    double cy = std::cos(e[1]);
    double sz = std::sin(e[2]);
    double cz = std::cos(e[2]);

    std::array<double, 9> R;
    R[0] = cy * cz;
    R[1] = sx * sy * cz + cx * sz;
    R[2] = -cx * sy * cz + sx * sz;
    R[3] = -cy * sz;
    R[4] = -sx * sy * sz + cx * cz;
    R[5] = cx * sy * sz + sx * cz;
    R[6] = sy;
    R[7] = -sx * cy;
    R[8] = cx * cy;

    return isometry(R, t);
  }

  inline void transform(point_3d& p) const {
    double rx = p[0] * R_[0] + p[1] * R_[3] + p[2] * R_[6];
    double ry = p[0] * R_[1] + p[1] * R_[4] + p[2] * R_[7];
    double rz = p[0] * R_[2] + p[1] * R_[5] + p[2] * R_[8];
    p[0] = rx + t_[0];
    p[1] = ry + t_[1];
    p[2] = rz + t_[2];
  }

 private:
  std::array<double, 9> const R_;
  std::array<double, 3> const t_;
};

class uosr_scan_reader {
 public:
  uosr_scan_reader(std::filesystem::path const& path_3d)
      : pose_(read_pose(
            std::filesystem::path(path_3d).replace_extension(string_pose))),
        stream_(internal::open_stream(path_3d.string(), std::ios::in)) {}

  inline bool read_next(point_3d& point) {
    std::string line;

    if (!std::getline(stream_, line)) {
      return false;
    }

    std::string value;
    char* end;
    std::stringstream ss(line);
    ss >> value;
    point[0] = std::strtod(value.c_str(), &end);
    ss >> value;
    point[1] = std::strtod(value.c_str(), &end);
    ss >> value;
    point[2] = std::strtod(value.c_str(), &end);
    // Fourth value would be the reflectance.
    // ss >> value;
    // float val = std::strtof(value.c_str(), &end);

    pose_.transform(point);

    return true;
  }

 private:
  isometry read_pose(std::filesystem::path const& path) {
    std::fstream stream = internal::open_stream(path.string(), std::ios::in);
    std::array<double, 3> e, t;
    stream >> t[0] >> t[1] >> t[2] >> e[0] >> e[1] >> e[2];

    e[0] *= deg_to_rad;
    e[1] *= deg_to_rad;
    e[2] *= deg_to_rad;

    return isometry::from_euler(e, t);
  }

  isometry const pose_;
  std::fstream stream_;
};

}  // namespace internal

template <typename Scalar_>
inline void read_uosr(
    std::string const& root, std::vector<point<Scalar_, 3>>& points) {
  if (!std::filesystem::exists(root)) {
    std::cout << "Path doesn't exist: " << root << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!std::filesystem::is_directory(root)) {
    std::cout << "Path isn't a directory: " << root << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // This doesn't test if the corresponding .pose exists, that is handled by the
  // uosr_scan_reader.
  std::regex pattern(
      std::string("^") + internal::string_scan + "\\d{3}" +
      internal::string_3d + "$");
  std::vector<std::filesystem::path> paths;
  for (auto const& entry : std::filesystem::directory_iterator(root)) {
    if (std::filesystem::is_regular_file(entry) &&
        std::regex_search(entry.path().filename().string(), pattern)) {
      paths.push_back(entry.path());
    }
  }

  point_3d point;
  for (auto const& path : paths) {
    // Printing the string doesn't show escape characters.
    std::cout << "Reading from scan: " << path.string() << std::endl;
    internal::uosr_scan_reader reader(path);
    while (reader.read_next(point)) {
      points.push_back(point.cast<Scalar_>());
    }
  }
}

}  // namespace pico_tree
