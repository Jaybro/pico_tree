# PicoTree

[![build-and-test](https://github.com/Jaybro/pico_tree/workflows/build-and-test/badge.svg)](https://github.com/Jaybro/pico_tree/actions?query=workflow%3Abuild-and-test) [![pip](https://github.com/Jaybro/pico_tree/workflows/pip/badge.svg)](https://github.com/Jaybro/pico_tree/actions?query=workflow%3Apip)

PicoTree is a C++ header only library with [Python bindings](https://github.com/pybind/pybind11) for fast nearest neighbor searches and range searches using a KdTree. See the table below to get an impression of the performance provided by the [KdTree](https://en.wikipedia.org/wiki/K-d_tree) of this library versus several other implementations:

|                                     | Build C++ | Build Python  | Knn C++    | Knn Python  |
| ----------------------------------- | --------- | ------------- | ---------- | ----------- |
| [nanoflann][nano] v1.5.0            | 2.9s      | ...           | 3.2s       | ...         |
| [SciPy KDTree][spkd] 1.11.0         | ...       | 4.5s          | ...        | 563.6s      |
| [Scikit-learn KDTree][skkd] 1.2.2   | ...       | 6.2s          | ...        | 42.2s       |
| [pykdtree][pykd] 1.3.7              | ...       | 1.0s          | ...        | 6.6s        |
| [OpenCV FLANN][cvfn] 4.6.0          | 1.9s      | ...           | 4.7s       | ...         |
| PicoTree KdTree v0.8.0              | 0.9s      | 1.0s          | 2.8s       | 3.1s        |

Two [LiDAR](./docs/benchmark.md) based point clouds of sizes 7733372 and 7200863 were used to generate these numbers. The first point cloud was the input to the build algorithm and the second to the query algorithm. All benchmarks were run on a single thread with the following parameters: `max_leaf_size=10` and `knn=1`. A more detailed [C++ comparison](./docs/benchmark.md) of PicoTree is available with respect to [nanoflann][nano].

[nano]: https://github.com/jlblancoc/nanoflann
[spkd]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
[skkd]: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
[pykd]: https://github.com/storpipfugl/pykdtree
[cvfn]: https://github.com/opencv/opencv

Available under the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.

# Capabilities

KdTree:
* Nearest neighbor, approximate nearest neighbor, radius, box, and customizable nearest neighbor searches.
* [Metrics](https://en.wikipedia.org/wiki/Metric_(mathematics)):
  * Support for topological spaces with identifications. E.g., points on the circle `[-pi, pi]`.
  * Available metrics: `L1`, `L2Squared`, `SO2`, and `SE2Squared`. Metrics can be customized.
* Multiple tree splitting rules: `kLongestMedian`, `kMidpoint` and `kSlidingMidpoint`.
* Compile time and run time known dimensions.
* Static tree builds.
* Thread safe queries.
* Optional [Python bindings](https://github.com/pybind/pybind11).

PicoTree can interface with different types of points and point sets through traits classes. These can be custom implementations or one of the `pico_tree::SpaceTraits<>` and `pico_tree::PointTraits<>` classes provided by this library.
* Space type support:
  * `std::vector<PointType>`.
  * `pico_tree::SpaceMap<PointType>`.
  * `Eigen::Matrix<>` and `Eigen::Map<Eigen::Matrix<>>`.
  * `cv::Mat`.
* Point type support:
  * Fixed size arrays and `std::array<>`.
  * `pico_tree::PointMap<>`.
  * `Eigen::Vector<>` and `Eigen::Map<Eigen::Vector<>>`.
  * `cv::Vec<>`.
* `pico_tree::SpaceMap<PointType>` and `pico_tree::PointMap<>` allow interfacing with dynamic size arrays. It is assumed that points and their coordinates are laid out contiguously in memory.

# Examples

* [Minimal working example](./examples/kd_tree/kd_tree_minimal.cpp) using an `std::vector<>` of points.
* [Creating a KdTree](./examples/kd_tree/kd_tree_creation.cpp) and taking the input by value or reference.
* Using the KdTree's [search](./examples/kd_tree/kd_tree_search.cpp) capabilities.
* Working with [dynamic size arrays](./examples/kd_tree/kd_tree_dynamic_arrays.cpp).
* Supporting a [custom point type](./examples/kd_tree/kd_tree_custom_point_type.cpp).
* Supporting a [custom space type](./examples/kd_tree/kd_tree_custom_space_type.cpp).
* Creating a [custom search visitor](./examples/kd_tree/kd_tree_custom_search_visitor.cpp).
* [Saving and loading](./examples/kd_tree/kd_tree_save_and_load.cpp) a KdTree to and from a file.
* Support for [Eigen](./examples/eigen/eigen.cpp) and [OpenCV](./examples/opencv/opencv.cpp) data types.
* How to use the [KdTree with Python](./examples/python/kd_tree.py).

# Requirements

Minimum:

* A compiler that is compliant with the C++17 standard or higher.
* [CMake](https://cmake.org/). It is also possible to just copy and paste the [pico_tree](./src/pico_tree/) directory into an include directory.

Optional:

* [Doxygen](https://www.doxygen.nl). Needed for generating documentation.
* [Google Test](https://github.com/google/googletest). Used for running unit tests.
* [Eigen](http://eigen.tuxfamily.org). To run the example that shows how Eigen data types can be used in combination with PicoTree.
* [OpenCV](https://opencv.org/). To run the OpenCV example that shows how to work with OpenCV data types.
* [Google Benchmark](https://github.com/google/benchmark) is needed to run any of the benchmarks. The [nanoflann](https://github.com/jlblancoc/nanoflann) and [OpenCV](https://opencv.org/) benchmarks also require their respective libraries to be installed.

Python bindings:
* [Python](https://www.python.org/). Version 3.7 or higher.
* [pybind11](https://github.com/pybind/pybind11). Used to ease the creation of Python bindings. Available under the [BSD](https://github.com/pybind/pybind11/blob/master/LICENSE) license and copyright.
* [OpenMP](https://www.openmp.org/). For parallelization of queries.
* [numpy](https://numpy.org/). Points and search results are represented by ndarrays.
* [scikit-build](https://scikit-build.readthedocs.io/). Glue between CMake and setuptools.

# Build

Build with [CMake](https://cmake.org/):

```console
$ mkdir build && cd build
$ cmake ../
$ cmake --build .
$ cmake --build . --target pico_tree_doc
$ cmake --install .
```

```cmake
find_package(PicoTree REQUIRED)

add_executable(myexe main.cpp)
target_link_libraries(myexe PUBLIC PicoTree::PicoTree)
```

Install with [pip](https://pypi.org/project/pip/):

```console
$ pip install ./pico_tree
```

# References

* [Computational Geometry - Algorithms and Applications.](https://www.springer.com/gp/book/9783540779735) Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars, Springer-Verlag, third edition, 2008.
* S. Maneewongvatana and D. M. Mount. [It's okay to be skinny, if your friends are fat.](http://www.cs.umd.edu/~mount/Papers/cgc99-smpack.pdf) 4th Annual CGC Workshop on Computational Geometry, 1999.
* S. Arya and H. Y. Fu. [Expected-case complexity of approximate nearest neighbor searching.](https://www.cse.ust.hk/faculty/arya/pub/exp.pdf) InProceedings of the 11th ACM-SIAM Symposium on Discrete Algorithms, 2000.
* S. Arya and D. M. Mount. [Algorithms for fast vector quantization.](https://www.cs.umd.edu/~mount/Papers/DCC.pdf) In IEEE Data Compression Conference, pages 381–390, March 1993.
* N. Sample, M. Haines, M. Arnold and T. Purcell. [Optimizing Search Strategies in k-d Trees.](http://infolab.stanford.edu/~nsample/pubs/samplehaines.pdf) In: 5th WSES/IEEE World Multiconference on Circuits, Systems, Communications & Computers (CSCC 2001), July 2001.
* A. Yershova and S. M. LaValle, [Improving Motion-Planning Algorithms by Efficient Nearest-Neighbor Searching.](http://msl.cs.uiuc.edu/~lavalle/papers/YerLav06.pdf) In IEEE Transactions on Robotics, vol. 23, no. 1, pp. 151-157, Feb. 2007.
