#!/usr/bin/env python3

import pico_tree as pt
import numpy as np


def tree_creation_and_query_types():
    p = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
    # In and output distances are squared distances when using Metric.L2.
    t = pt.KdTree(p, pt.Metric.L2, 1)
    print(f"{t}")
    print(f"Number of points used to build the tree: {t.npts()}")
    print(f"Spatial dimension of the tree: {t.sdim()}")
    value = -2.0
    print(f"Metric applied to {value}: {t.metric(value)}")

    # Nearest neighbors via return.
    nns = t.search_knns(p, 1)
    print("Single nn for each input point:")
    print(nns)
    # Possibly re-use the memory in a another query.
    # If the input size is incorrect, it gets resized.
    t.search_knns(p, 2, nns)
    print("Two nns for each input point:")
    print(nns)

    # Searching for approximate nearest neighbors works the same way.
    # An approximate nearest neighbor can be at most a distance factor of 1+e
    # farther away from the true nearest neighbor.
    max_error = 0.75
    # Apply the metric function to the ratio to get the squared ratio.
    max_error_ratio = t.metric(1.0 + max_error)
    nns = t.search_aknns(p, 2, max_error_ratio)
    t.search_aknns(p, 2, max_error_ratio, nns)
    # Note that we scale back the ann distance its original distance.
    print("The 2nd closest to each input point:")
    for r in nns:
        print(
            f"Point index {r[1][0]} with distance {r[1][1] * max_error_ratio}")

    # A radius search doesn't return a numpy array but a custom vector of numpy
    # arrays. This is because the amount of neighbors to each of input points
    # may vary for a radius search.
    search_radius = t.metric(2.5)
    print(f"Result with radius: {search_radius}")
    rns = t.search_radius(p, search_radius)
    for neighborhood in rns:
        print(f"{neighborhood}")
    search_radius = t.metric(5.0)
    print(f"Result with radius: {search_radius}")
    t.search_radius(p, 25.0, rns)
    for neighborhood in rns:
        print(f"{neighborhood}")

    # The custom type can also be indexed.
    print(f"Result size: {len(rns)}")
    # Note that each numpy array is actually a view of a C++ vector.
    print(f"First index: {rns[0]}")


def main():
    tree_creation_and_query_types()


if __name__ == "__main__":
    main()
