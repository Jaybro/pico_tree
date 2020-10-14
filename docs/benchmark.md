# Benchmark

One of the PicoTree examples is a small [benchmark](./examples/benchmark/) that compares the KdTree of this library with that of [nanoflann](https://github.com/jlblancoc/nanoflann). This page describes the benchmark output and how to reproduce the exact same input that was used for benchmarking.

# Data sets

The [Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/) provides several 3D point clouds. The following two have been used for the comparison benchmark:

* #25 - Würzburg marketplace. Authors: Johannes Schauer, Andreas Nüchter from the University of Würzburg, Germany.
* #26 - Maria-Schmerz-Kapelle Randersacker. Authors: Andreas Nüchter, Helge Andreas Lauterbach from the University of Würzburg, Germany.

Both have been generated using a LiDAR scanner and represent different types of environments. The running time of the benchmark was kept reasonable by using a subset of points and storing those in a simple binary format. The final point cloud sizes were as follows:

* #25 - Würzburg marketplace: 13729039 points.
* #26 - Maria-Schmerz-Kapelle Randersacker: 20793160 points.

# Results

The different KdTree implementations are compared to each other with respect to the running times of the build, radius search and knn search algorithms, while fixing certain parameters. The speed of each algorithm is plotted against the leaf size of the tree. Each algorithm sets the following parameters:

* Build algorithm: Dimensions known at compile time or run time.
* Radius search algorithm: The radius in meters divided by 4 (0.25m and 0.5m).
* Knn algorithm: The mount of neighbors searched.

Note that the run time describes a single invocation of the build algorithm and n invocations of the others.

Results were generated on: 20-09-2020 using MinGW GCC 10.1

For a "special" case of the knn algorithm, where `k` is set to `1`, the search speed is compared with respect to different tree building techniques:
* Nanoflann Midpoint variation.
* PicoTree Sliding Midpoint (along the longest axis).
* PicoTree Longest Axis Median.

It is interesting to see that finding a single nearest neighbor can be quite a bit faster using the Longest Axis Median splitting technique. However, building the tree or querying multiple neighbors is slower. The extra time it takes to build the tree is no longer a factor when the tree is queried multiple times (on the test sets this was about `2n-4n` times). This means the splitting technique can be useful in combination with an algorithm like [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point).

Results were generated on: 01-10-2020 using MinGW GCC 10.1

## #25 - Würzburg marketplace.

![Square Build Time](./images/benchmark_square_build_time.png)![Square Radius Search Time](./images/benchmark_square_radius_search_time.png)

![Square Knn Search Time](./images/benchmark_square_knn_search_time.png)![Square Knn1 Search Time](./images/benchmark_square_knn1_search_time.png)

## #26 - Maria-Schmerz-Kapelle Randersacker.

![Square Build Time](./images/benchmark_church_build_time.png)![Square Radius Search Time](./images/benchmark_church_radius_search_time.png)

![Square Knn Search Time](./images/benchmark_church_knn_search_time.png)![Square Knn1 Search Time](./images/benchmark_church_knn1_search_time.png)

# Running a new benchmark

The following steps can be taken to reproduce the data sets:

1. Download and unpack a data set. This results in a directory containing several pairs of `.3d` and `.pose` files, each representing a LiDAR scan.
2. Delete all scans but the first one (only keeping `scan000.3d` and `scan000.pose`).
3. Run the `uosr_to_bin` executable as a sibling to the scan to generate a `scans.bin` file.

To get performance statistics:

4. Run the `benchmark` executable as a sibling to the `scans.bin` file and set the output format to `json`.
5. Run `plot_benchmarks.py` to show and store the performance plots.

Note the following:

* The `uosr_to_bin` tool simply combines all pairs of `.3d` and `.pose` files into a single `.bin` file. This means a bigger point cloud can be used for benchmarking.
* A `.txt` file can be generated from the `.bin` file by running the `bin_to_ascii` executable (as a sibling to the binary file). Each line in the output file is a 3D point.
