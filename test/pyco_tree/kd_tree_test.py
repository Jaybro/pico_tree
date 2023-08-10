#!/usr/bin/env python3

import unittest
import numpy as np
import pico_tree as pt


class KdTreeTest(unittest.TestCase):
    def test_creation_kd_tree(self):
        # Row major input check.
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float64, order='C')
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        self.assertEqual(a.shape[0], t.npts)
        self.assertEqual(a.shape[1], t.sdim)
        # The scalar dtype of the tree should be the same as the input.
        self.assertEqual(a.dtype, t.dtype_scalar)

        # Non-contiguous arrays are not supported.
        a = a[::2]
        with self.assertRaises(RuntimeError):
            t = pt.KdTree(a, pt.Metric.L2Squared, 10)

        # Col major input check.
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32, order='F')
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        self.assertEqual(a.shape[1], t.npts)
        self.assertEqual(a.shape[0], t.sdim)
        self.assertEqual(a.dtype, t.dtype_scalar)

        # The tree implements the buffer protocol and can be inspected using a
        # memoryview. We'll use the view to check that the tree didn't copy the
        # numpy array.
        # Note: This invalidates the built index by the tree.
        a[0][0] = 42
        self.assertEqual(a[0][0], memoryview(t)[0, 0])

        # The tree must have a dimension of two.
        with self.assertRaises(TypeError):
            a = np.array([[[2, 1]], [[4, 3]], [[8, 7]]], dtype=np.float32)
            t = pt.KdTree(a, pt.Metric.L2Squared, 10)

    def test_metric(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        # L2Squared
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        self.assertEqual(t.metric(-2.0), 4)
        # L1
        t = pt.KdTree(a, pt.Metric.L1, 10)
        self.assertEqual(t.metric(-2.0), 2)

    def test_search_knn(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        k = 2

        # Test if the query actually works
        nns = t.search_knn(a, k)
        self.assertEqual(nns.shape, (3, k))

        for i in range(len(nns)):
            self.assertEqual(nns[i][0][0], i)
            self.assertAlmostEqual(nns[i][0][1], 0)

        # Test that the memory is re-used
        data = nns.ctypes.data
        t.search_knn(a, k, nns)
        self.assertEqual(nns.ctypes.data, data)

    def test_search_approximate_knn(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        k = 2
        s = 1.0

        # Test if the query actually works
        nns = t.search_knn(a, k, s)
        self.assertEqual(nns.shape, (3, k))

        for i in range(len(nns)):
            self.assertEqual(nns[i][0][0], i)
            self.assertAlmostEqual(nns[i][0][1], 0)

        # Test that the memory is re-used
        data = nns.ctypes.data
        t.search_knn(a, k, s, nns)
        self.assertEqual(nns.ctypes.data, data)

    def test_search_radius(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)

        radius = t.metric(2.5)
        nns = t.search_radius(a, radius)
        self.assertEqual(len(nns), 3)
        self.assertEqual(nns.dtype, t.dtype_neighbor)
        self.assertTrue(nns)

        for i, n in enumerate(nns):
            self.assertEqual(len(n), 1)
            self.assertEqual(n[0][0], i)
            self.assertAlmostEqual(n[0][1], 0)

        # This checks if DArray is also a sequence.
        for i in range(len(nns)):
            self.assertEqual(nns[i][0][0], i)
            self.assertAlmostEqual(nns[i][0][1], 0)

        # Test that the memory is re-used by comparing memory addresses.
        # In case the size of an array equals zero, its memory address is
        # random. See darray.hpp for more details.
        def addresses(nns):
            return [x.ctypes.data if len(x) else 0 for x in nns]

        datas = addresses(nns)
        t.search_radius(a, radius, nns)
        self.assertEqual(addresses(nns), datas)

    def test_search_approximate_radius(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        s = 1.0

        radius = t.metric(2.5)
        nns = t.search_radius(a, radius, s)
        self.assertEqual(len(nns), 3)
        self.assertEqual(nns.dtype, t.dtype_neighbor)
        self.assertTrue(nns)

        for i, n in enumerate(nns):
            self.assertEqual(len(n), 1)
            self.assertEqual(n[0][0], i)
            self.assertAlmostEqual(n[0][1], 0)

        # This checks if DArray is also a sequence.
        for i in range(len(nns)):
            self.assertEqual(nns[i][0][0], i)
            self.assertAlmostEqual(nns[i][0][1], 0)

        # Test that the memory is re-used by comparing memory addresses.
        # In case the size of an array equals zero, its memory address is
        # random. See darray.hpp for more details.
        def addresses(nns):
            return [x.ctypes.data if len(x) else 0 for x in nns]

        datas = addresses(nns)
        t.search_radius(a, radius, s, nns)
        self.assertEqual(addresses(nns), datas)

    def test_search_box(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        boxes = np.array(
            [[0, 0],
             [3, 3],
             [2, 2],
             [3, 3],
             [0, 0],
             [9, 9],
             [6, 6],
             [9, 9]],
            dtype=np.float32)
        nns = t.search_box(boxes)
        self.assertEqual(len(nns), 4)
        self.assertEqual(nns.dtype, t.dtype_index)
        self.assertTrue(nns)

        # Test that the memory is re-used by comparing memory addresses.
        # In case the size of an array equals zero, its memory address is
        # random. See darray.hpp for more details.
        def addresses(nns):
            return [x.ctypes.data if len(x) else 0 for x in nns]

        datas = addresses(nns)
        t.search_box(boxes, nns)
        self.assertEqual(addresses(nns), datas)

        # Check the number of indices found.
        sizes = [1, 0, 3, 1]
        for n, s in zip(nns, sizes):
            self.assertEqual(len(n), s)

        nns = nns[0:4:2]
        self.assertEqual(len(nns), 2)

        sizes = [1, 3]
        for n, s in zip(nns, sizes):
            self.assertEqual(len(n), s)

        # Test negative indexing.
        self.assertEqual(len(nns[-1]), 3)

    def test_creation_darray(self):
        # A DArray can be created from 3 different dtypes or their descriptors.
        # It is the easiest to use the dtype properties of the KdTree.

        # 1) [('index', '<i4'), ('distance', '<f4')]
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        d = pt.DArray(t.dtype_neighbor)
        self.assertEqual(d.dtype, t.dtype_neighbor)
        self.assertFalse(d)

        # 2)
        #   {'names':['index','distance'],
        #    'formats':['<i4','<f8'],
        #    'offsets':[0,8], 'itemsize':16}
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float64)
        t = pt.KdTree(a, pt.Metric.L2Squared, 10)
        d = pt.DArray(dtype=t.dtype_neighbor)
        self.assertEqual(d.dtype, t.dtype_neighbor)
        self.assertFalse(d)

        # 3) int32
        d = pt.DArray(np.int32)
        self.assertEqual(d.dtype, t.dtype_index)
        d = pt.DArray(np.dtype(np.int32))
        self.assertEqual(d.dtype, t.dtype_index)
        self.assertFalse(d)


if __name__ == '__main__':
    unittest.main()
