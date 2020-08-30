#pragma once

#include <point.hpp>

namespace pico_tree {

void WriteBin(std::string const& filename, std::vector<Point3d> const& points);

std::vector<Point3d> ReadBin(std::string const& filename);

}  // namespace pico_tree
