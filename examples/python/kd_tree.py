#!/usr/bin/env python3

import pico_tree as pt
import numpy as np
from pathlib import Path
from time import perf_counter
# from scipy.spatial import KDTree as spKDTree
# from sklearn.neighbors import KDTree as skKDTree
# from pykdtree.kdtree import KDTree as pyKDTree


def tree_creation_and_query_types():
    print("*** KdTree Creation And Basic Information ***")
    # An input array must have a dimension of two and it must be contiguous. A
    # C contiguous array contains points in its rows and an F contiguous array
    # contains points in its columns.
    p = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
    # Both the in and output distances are squared when using Metric.L2Squared.
    t = pt.KdTree(p, pt.Metric.L2Squared, 1)
    print(f"{t}")
    print(f"Number of points used to build the tree: {t.npts}")
    print(f"Spatial dimension of the tree: {t.sdim}")
    value = -2.0
    print(f"Metric applied to {value}: {t.metric(value)}")
    print()

    print("*** Nearest Neighbor Search ***")
    # Nearest neighbors via return.
    knns = t.search_knn(p, 1)
    print("Single nn for each input point:")
    print(knns)
    # Possibly re-use the memory in a another query.
    # If the input size is incorrect, it gets resized.
    t.search_knn(p, 2, knns)
    print("Two nns for each input point:")
    print(knns)
    print()

    print("*** Approximate Nearest Neighbor Search ***")
    # Searching for approximate nearest neighbors works the same way.
    # An approximate nearest neighbor can be at most a distance factor of 1+e
    # farther away from the true nearest neighbor.
    max_error = 0.75
    # Apply the metric function to the ratio to get the squared ratio.
    max_error_ratio = t.metric(1.0 + max_error)
    knns = t.search_knn(p, 2, max_error_ratio)
    t.search_knn(p, 2, max_error_ratio, knns)
    # Note that we scale back the ann distance its original distance.
    print("The 2nd closest to each input point:")
    for knn in knns:
        print(
            f"Point index {knn[1][0]} with distance {knn[1][1] * max_error_ratio}")
    print()

    print("*** Radius Search ***")
    # A radius search doesn't return a numpy array but a custom vector of numpy
    # arrays. This is because the amount of neighbors to each of input points
    # may vary for a radius search.
    search_radius = t.metric(2.5)
    print(f"Result with radius: {search_radius}")
    rnns = t.search_radius(p, search_radius)
    for rnn in rnns:
        print(f"{rnn}")
    search_radius = t.metric(5.0)
    t.search_radius(p, 25.0, rnns)
    print(f"Result with radius: {search_radius}")
    for rnn in rnns:
        print(f"{rnn}")
    print()

    print("*** Box Search ***")
    # A box search returns the same data structure as a radius search. However,
    # instead of containing neighbors it simply contains indices.
    min = np.array([[0, 0], [2, 2], [0, 0], [6, 6]], dtype=np.float32)
    max = np.array([[3, 3], [3, 3], [9, 9], [9, 9]], dtype=np.float32)
    bnns = t.search_box(min, max)
    t.search_box(min, max, bnns)
    print("Results for the orthogonal box search:")
    for bnn in bnns:
        print(f"{bnn}")
    print()

    print("*** DArray ***")
    # The custom type can also be indexed.
    print(f"Result size: {len(bnns)}")
    # Note that each numpy array is actually a view of a C++ vector.
    print(f"First index: {bnns[0]}")
    print(f"Second last index: {bnns[-2]}")
    half = bnns[0:4:2]
    print("Sliced results for the orthogonal box search:")
    for bnn in half:
        print(f"{bnn}")
    print()


def array_initialization():
    print("*** Array Initialization ***")
    p = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float64)
    # In and output distances are absolute distances when using Metric.L1.
    t = pt.KdTree(p, pt.Metric.L1, 10)

    # This type of forward initialization of arrays may be useful to streamline
    # loops that depend on them and where reusing memory is desired. E.g.: ICP.
    knns = np.empty((0), dtype=t.dtype_neighbor)
    print(knns.dtype)
    rnns = pt.DArray(dtype=t.dtype_neighbor)
    print(rnns.dtype)
    bnns = pt.DArray(dtype=t.dtype_index)
    print(bnns.dtype)
    print()


def performance_test_pico_tree():
    print("*** Performance against scans.bin ***")
    # The benchmark documentation, docs/benchmark.md section "Running a new
    # benchmark", explains how to generate a scans.bin file from an online
    # dataset.
    try:
        p0 = np.fromfile(Path(__file__).parent / "scans0.bin",
                         np.float32).reshape((-1, 3))
        p1 = np.fromfile(Path(__file__).parent / "scans1.bin",
                         np.float32).reshape((-1, 3))
    except FileNotFoundError as e:
        print(f"Skipping test. File does not exist: {e.filename}")
        return

    cnt_build_time_before = perf_counter()
    # Tree creation is only slightly slower in Python vs C++ using the bindings.
    t = pt.KdTree(p0, pt.Metric.L2Squared, 10)
    #t = spKDTree(p0, leafsize=10)
    #t = skKDTree(p0, leaf_size=10)
    #t = pyKDTree(p0, leafsize=10)
    cnt_build_time_after = perf_counter()
    print(f"{t} was built in {(cnt_build_time_after - cnt_build_time_before) * 1000.0}ms")
    # Use the OMP_NUM_THREADS environment variable to influence the number of
    # threads used for querying: export OMP_NUM_THREADS=1
    k = 1
    cnt_query_time_before = perf_counter()
    # Searching for nearest neighbors is a constant amount of time slower
    # using the bindings as compared to the C++ benchmark (regardless of k).
    # The following must be noted however: The Python benchmark simply calls
    # the knn function provided by the Python bindings. As such it does not
    # directly wrap the C++ benchmark. This means the performance difference is
    # not only due to the bindings overhead. The C++ implementation benchmark
    # may have been optimized more because is very simple. The bindings also
    # have various extra overhead: checks, numpy array memory creation, OpenMP,
    # etc.
    # TODO The actual overhead is probably very similar to that of the KdTree
    # creation, but it would be nice to measure the overhead w.r.t. the actual
    # query.
    unused_knns = t.search_knn(p1, k)
    #unused_dd, unused_ii = t.query(p1, k=k)
    cnt_query_time_after = perf_counter()
    print(
        f"{len(p1)} points queried in {(cnt_query_time_after - cnt_query_time_before) * 1000.0}ms")
    print()


def main():
    tree_creation_and_query_types()
    array_initialization()
    performance_test_pico_tree()


if __name__ == "__main__":
    main()
