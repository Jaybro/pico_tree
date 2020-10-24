# PicoTree

[![build-and-test](https://github.com/Jaybro/pico_tree/workflows/build-and-test/badge.svg)](https://github.com/Jaybro/pico_tree/actions?query=workflow%3Abuild-and-test)

PicoTree is a small C++ header only library for range searches and nearest neighbor searches using a KdTree.

See the comparison [benchmark](./docs/benchmark.md) between PicoTree and [nanoflann](https://github.com/jlblancoc/nanoflann) to get an impression of the performance provided by the [KdTree](https://en.wikipedia.org/wiki/K-d_tree) of this library.

Available under the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.

# Capabilities

* KdTree
  * Nearest neighbors, radius, and box searches.
  * Customizable nearest neighbor searches, [metrics](https://en.wikipedia.org/wiki/Metric_(mathematics)) and tree splitting techniques.
  * Compile time and run time known dimensions.
  * Static tree builds.

The examples show how PicoTree can be used:

* Creating an [adaptor](./examples/pico_common/pico_adaptor.hpp) to interface with input point clouds.
* Searching using the [KdTree](./examples/kd_tree/kd_tree.cpp) and creating a custom search visitor (for finding approximate nearest neighbors).
* Using [Eigen](./examples/eigen/) data types.

# Requirements

Minimum:

* A compiler that is compliant with the C++11 standard or higher.
* [CMake](https://cmake.org/). It is also possible to simply copy and paste the contents of the [src](./src/) directory given that PicoTree is header only.

Optional:

* [Doxygen](https://www.doxygen.nl). Needed for generating documentation.
* [Google Test](https://github.com/google/googletest). Used for running unit tests.
* [Eigen](http://eigen.tuxfamily.org). To run the example that shows how Eigen data types can be used in combination with PicoTree.
* [nanoflann](https://github.com/jlblancoc/nanoflann), [Google Benchmark](https://github.com/google/benchmark) and a compiler that is compliant with the C++17 standard are needed to run the comparison [benchmark](./docs/benchmark.md) between nanoflann and PicoTree.

# Build

An example using [CMake](https://cmake.org/) with [MSYS2](https://github.com/msys2/) and [MinGW64](http://mingw-w64.org/):

```console
$ mkdir build && cd build
$ cmake.exe ../ -G "MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=C:/msys64/mingw64/local
$ mingw32-make.exe
$ mingw32-make.exe install
$ mingw32-make.exe pico_tree_doc
```

```cmake
find_package(PicoTree REQUIRED)

add_executable(myexe main.cpp)
target_link_libraries(myexe PUBLIC PicoTree::PicoTree)
```

# References

* [Computational Geometry - Algorithms and Applications.](https://www.springer.com/gp/book/9783540779735) Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars, Springer-Verlag, third edition, 2008.
* S. Maneewongvatana and D. M. Mount. [It's okay to be skinny, if your friends are fat.](http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf) 4th Annual CGC Workshop on Computational Geometry, 1999.
* https://en.wikipedia.org/wiki/K-d_tree
