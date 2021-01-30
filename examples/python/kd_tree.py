#!/usr/bin/env python3

import pico_tree as pt
import numpy as np


def tree_creation_and_query_types():
    print("*** KdTree Creation And Basic Information ***")
    p = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
    # In and output distances are squared distances when using Metric.L2.
    t = pt.KdTree(p, pt.Metric.L2, 1)
    print(f"{t}")
    print(f"Number of points used to build the tree: {t.npts()}")
    print(f"Spatial dimension of the tree: {t.sdim()}")
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
    knns = t.search_aknn(p, 2, max_error_ratio)
    t.search_aknn(p, 2, max_error_ratio, knns)
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

    # The custom type can also be indexed.
    print(f"Result size: {len(rnns)}")
    # Note that each numpy array is actually a view of a C++ vector.
    print(f"First index: {rnns[0]}")
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


def main():
    tree_creation_and_query_types()


if __name__ == "__main__":
    main()
