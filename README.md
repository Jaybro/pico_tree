# PicoTree

[![build-and-test](https://github.com/Jaybro/pico_tree/workflows/build-and-test/badge.svg)](https://github.com/Jaybro/pico_tree/actions?query=workflow%3Abuild-and-test) [![pip](https://github.com/Jaybro/pico_tree/workflows/pip/badge.svg)](https://github.com/Jaybro/pico_tree/actions?query=workflow%3Apip)

PicoTree is a C++ header only library with [Python bindings](https://github.com/pybind/pybind11) for range searches and nearest neighbor searches using a KdTree.

See the table below to get an impression of the performance provided by the [KdTree](https://en.wikipedia.org/wiki/K-d_tree) of this library versus several other implementations:

|                             | Build C++ | Build Python  | Knn C++   | Knn Python  |
| --------------------------- | --------- | ------------- | ----------| ----------- |
| [nanoflann][nano]           | 3.6s      | ...           | 5.2s      | ...         |
| [SciPy KDTree][sppk]        | ...       | 117.9s        | ...       | +inf        |
| [SciPy cKDTree][spck]       | ...       | 9.6s          | ...       | 14.1s       |
| [Scikit-learn KDTree][skck] | ...       | 27.1s         | ...       | 55.4s       |
| PicoTree                    | 2.0s      | 2.1s          | 3.1s      | 4.1s        |

The [comparison](./examples/python/kd_tree.py) was generated using 13729039 3d points from a [LiDAR](./docs/benchmark.md) scan, float64, with k = 1 for queries. Note that the Python Knn benchmark does not directly wrap the C++ Knn benchmark, meaning that they can't be compared directly. A more detailed [comparison](./docs/benchmark.md) of PicoTree is available with respect to [nanoflann][nano].

Available under the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.

[nano]: https://github.com/jlblancoc/nanoflann
[sppk]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
[spck]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
[skck]: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

# Capabilities

* KdTree
  * Nearest neighbors, approximate nearest neighbors, radius, and box searches.
  * Customizable nearest neighbor searches, [metrics](https://en.wikipedia.org/wiki/Metric_(mathematics)) and tree splitting techniques.
  * Compile time and run time known dimensions.
  * Static tree builds.
  * Thread safe queries.

The examples show how PicoTree can be used:

* How to create the expected [point](./examples/pico_common/point.hpp) and [point set](./examples/pico_common/pico_adaptor.hpp) interfaces.
* Searching using the [KdTree](./examples/kd_tree/kd_tree.cpp) and creating a custom search visitor.
* Support for [Eigen](./examples/eigen/eigen.cpp) data types.
* Using the [KdTree with Python](./examples/python/kd_tree.py).

# Requirements

Minimum:

* A compiler that is compliant with the C++11 standard or higher.
* [CMake](https://cmake.org/). It is also possible to just copy and paste the [pico_tree](./src/pico_tree/) directory into an include directory.

Optional:

* [Doxygen](https://www.doxygen.nl). Needed for generating documentation.
* [Google Test](https://github.com/google/googletest). Used for running unit tests.
* [Eigen](http://eigen.tuxfamily.org). To run the example that shows how Eigen data types can be used in combination with PicoTree.
* [nanoflann](https://github.com/jlblancoc/nanoflann), [Google Benchmark](https://github.com/google/benchmark) and a compiler that is compliant with the C++17 standard are needed to run the comparison [benchmark](./docs/benchmark.md) between nanoflann and PicoTree.

Python bindings:
* [Python](https://www.python.org/). Version 3.5 or higher.
* [pybind11](https://github.com/pybind/pybind11). Used to ease the creation of Python bindings. Available under the [BSD](https://github.com/pybind/pybind11/blob/master/LICENSE) license and copyright.
* [OpenMP](https://www.openmp.org/). For parallelization of queries.
* [numpy](https://numpy.org/). Points and search results are represented by ndarrays.
* [scikit-build](https://scikit-build.readthedocs.io/). Glue between CMake and setuptools.

# Build

Build using [CMake](https://cmake.org/):

```console
$ mkdir build && cd build
$ cmake ../
$ make
$ make install
$ make pico_tree_doc
```

Similarly with [MSYS2](https://github.com/msys2/) and [MinGW64](http://mingw-w64.org/):

```console
$ ...
$ cmake.exe ../ -G "MinGW Makefiles" -DCMAKE_INSTALL_PREFIX=C:/msys64/mingw64/local
$ mingw32-make.exe
$ ...
```

```cmake
find_package(PicoTree REQUIRED)

add_executable(myexe main.cpp)
target_link_libraries(myexe PUBLIC PicoTree::PicoTree)
```

Install with pip3:

```console
$ pip3 install ./pico_tree
```

Set a generator for use with MinGW:

```console
$ pip3 install ./pico_tree --install-option="-GMinGW Makefiles"
```

# References

* [Computational Geometry - Algorithms and Applications.](https://www.springer.com/gp/book/9783540779735) Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars, Springer-Verlag, third edition, 2008.
* S. Maneewongvatana and D. M. Mount. [It's okay to be skinny, if your friends are fat.](http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf) 4th Annual CGC Workshop on Computational Geometry, 1999.
* S. Arya and H. Y. Fu. [Expected-case complexity of approximate nearest neighbor searching.](https://www.cse.ust.hk/faculty/arya/pub/exp.pdf) InProceedings of the 11th ACM-SIAM Symposium on Discrete Algorithms, 2000.
* https://en.wikipedia.org/wiki/K-d_tree
