#!/usr/bin/env python3

import unittest
import numpy as np
import pico_tree as pt


class KdTreeTest(unittest.TestCase):
    def test_creation(self):
        # Row major
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32, order='C')
        t = pt.KdTree(a, pt.Metric.L2, 10)
        self.assertEqual(a.shape[0], t.npts)
        self.assertEqual(a.shape[1], t.sdim)
        # Col major
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32, order='F')
        t = pt.KdTree(a, pt.Metric.L2, 10)
        self.assertEqual(a.shape[1], t.npts)
        self.assertEqual(a.shape[0], t.sdim)
        # Other
        self.assertEqual(a.dtype, t.dtype_scalar)
        # No copy
        a[0][0] = 22
        self.assertEqual(a[0][0], memoryview(t)[0, 0])

    def test_metric(self):
        a = np.array([[2, 1], [4, 3], [8, 7]], dtype=np.float32, order='C')
        # L2
        t = pt.KdTree(a, pt.Metric.L2, 10)
        self.assertEqual(t.metric(-2.0), 4)
        # L1
        t = pt.KdTree(a, pt.Metric.L1, 10)
        self.assertEqual(t.metric(-2.0), 2)


if __name__ == '__main__':
    unittest.main()
