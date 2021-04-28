#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pico_toolshed/point.hpp>
#include <pico_tree/internal/stream.hpp>
#include <regex>
#include <string>

namespace pico_tree {

namespace internal {

constexpr auto kScan = "scan";
constexpr auto kPose = ".pose";
constexpr auto k3d = ".3d";
constexpr double kDegToRad = 3.1415926535897932384626433832795 / 180.0;

// Transformation that preserves distances between points. Column major.
class Isometry {
 public:
  Isometry(std::array<double, 9> const& R, std::array<double, 3> const& t)
      : R_{R}, t_{t} {}

  static Isometry FromEuler(
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

    return Isometry(R, t);
  }

  inline void Transform(Point3d* p) const {
    auto& d = *p;
    double rx = d(0) * R_[0] + d(1) * R_[3] + d(2) * R_[6];
    double ry = d(0) * R_[1] + d(1) * R_[4] + d(2) * R_[7];
    double rz = d(0) * R_[2] + d(1) * R_[5] + d(2) * R_[8];
    d(0) = rx + t_[0];
    d(1) = ry + t_[1];
    d(2) = rz + t_[2];
  }

 private:
  std::array<double, 9> const R_;
  std::array<double, 3> const t_;
};

class UosrScanReader {
 public:
  UosrScanReader(std::filesystem::path const& path_3d)
      : pose_(
            ReadPose(std::filesystem::path(path_3d).replace_extension(kPose))),
        stream_(internal::OpenStream(path_3d.string(), std::ios::in)) {}

  inline bool ReadNext(Point3d* point) {
    std::string line;

    if (!std::getline(stream_, line)) {
      return false;
    }

    std::string value;
    char* end;
    std::stringstream ss(line);
    ss >> value;
    (*point)(0) = std::strtod(value.c_str(), &end);
    ss >> value;
    (*point)(1) = std::strtod(value.c_str(), &end);
    ss >> value;
    (*point)(2) = std::strtod(value.c_str(), &end);
    // Fourth value would be the reflectance.
    // ss >> value;
    // float val = std::strtof(value.c_str(), &end);

    pose_.Transform(point);

    return true;
  }

 private:
  Isometry ReadPose(std::filesystem::path const& path) {
    std::fstream stream = internal::OpenStream(path.string(), std::ios::in);
    std::array<double, 3> e, t;
    stream >> t[0] >> t[1] >> t[2] >> e[0] >> e[1] >> e[2];

    e[0] *= kDegToRad;
    e[1] *= kDegToRad;
    e[2] *= kDegToRad;

    return Isometry::FromEuler(e, t);
  }

  Isometry const pose_;
  std::fstream stream_;
};

}  // namespace internal

template <typename Scalar_>
inline void ReadUosr(
    std::string const& root, std::vector<Point<Scalar_, 3>>* points) {
  if (!std::filesystem::exists(root)) {
    std::cout << "Path doesn't exist: " << root << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!std::filesystem::is_directory(root)) {
    std::cout << "Path isn't a directory: " << root << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // This doesn't test if the corresponding .pose exists, that is handled by the
  // UosrScanReader.
  std::regex pattern(
      std::string("^") + internal::kScan + "\\d{3}" + internal::k3d + "$");
  std::vector<std::filesystem::path> paths;
  for (auto const& entry : std::filesystem::directory_iterator(root)) {
    if (std::filesystem::is_regular_file(entry) &&
        std::regex_search(entry.path().filename().string(), pattern)) {
      paths.push_back(entry.path());
    }
  }

  Point3d point;
  for (auto const& path : paths) {
    // Printing the string doesn't show escape characters.
    std::cout << "Reading from scan: " << path.string() << std::endl;
    internal::UosrScanReader reader(path);
    while (reader.ReadNext(&point)) {
      points->push_back(point.Cast<Scalar_>());
    }
  }
}

}  // namespace pico_tree
