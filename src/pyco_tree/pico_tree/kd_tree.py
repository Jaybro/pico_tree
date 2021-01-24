import pico_tree._pyco_tree as _pt
from .metric import Metric

import numpy as np

__all__ = ['KdTree']


def _build_kd_tree(pts: np.array, metric: Metric, max_leaf_size: int, lut: dict):
    if pts.dtype != np.float32 and pts.dtype != np.float64:
        raise TypeError(
            "pts array format not supported: exptected float32 or float64")

    return lut[metric.value][pts.dtype.char](pts, max_leaf_size)


def _build_kd_tree_xd(pts: np.array, metric: Metric, max_leaf_size: int):
    lut = {Metric.L1.value: {'f': _pt.KdTreeXfL1,
                             'd': _pt.KdTreeXdL1},
           Metric.L2.value: {'f': _pt.KdTreeXfL2,
                             'd': _pt.KdTreeXdL2}}

    return _build_kd_tree(pts, metric, max_leaf_size, lut)


def _build_kd_tree_2d(pts: np.array, metric: Metric, max_leaf_size: int):
    lut = {Metric.L1.value: {'f': _pt.KdTree2fL1,
                             'd': _pt.KdTree2dL1},
           Metric.L2.value: {'f': _pt.KdTree2fL2,
                             'd': _pt.KdTree2dL2}}

    return _build_kd_tree(pts, metric, max_leaf_size, lut)


def _build_kd_tree_3d(pts: np.array, metric: Metric, max_leaf_size: int):
    lut = {Metric.L1.value: {'f': _pt.KdTree3fL1,
                             'd': _pt.KdTree3dL1},
           Metric.L2.value: {'f': _pt.KdTree3fL2,
                             'd': _pt.KdTree3dL2}}

    return _build_kd_tree(pts, metric, max_leaf_size, lut)


def KdTree(pts: np.array, metric: Metric, max_leaf_size: int):
    if pts.ndim != 2:
        raise TypeError("pts.ndim should equal 2")

    # All coordinates of a point have to be aligned. This in turn allows us to
    # determine the spatial dimension from either the row or col count.
    dim = pts.shape[0] if pts.flags["F_CONTIGUOUS"] else pts.shape[1]

    if dim == 2:
        return _build_kd_tree_2d(pts, metric, max_leaf_size)
    elif dim == 3:
        return _build_kd_tree_3d(pts, metric, max_leaf_size)
    else:
        return _build_kd_tree_xd(pts, metric, max_leaf_size)
