#pragma once

#include <fstream>

namespace pico_tree {

std::fstream OpenStream(
    std::string const& filename, std::ios_base::openmode mode);

}
