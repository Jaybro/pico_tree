#pragma once

#include <pico_toolshed/point.hpp>

namespace pico_tree {

std::vector<Point3d> ReadUosr(std::string const& directory);

}
