import pico_tree._pyco_tree as _pt
from .metric import Metric

import numpy as np
import numpy.typing as npt

__all__ = ['KdTree']

_lut_2d = {Metric.L1.value: {'f': _pt.KdTree2fL1,
                             'd': _pt.KdTree2dL1},
           Metric.L2Squared.value: {'f': _pt.KdTree2fL2Squared,
                                    'd': _pt.KdTree2dL2Squared},
           Metric.LInf.value: {'f': _pt.KdTree2fLInf,
                               'd': _pt.KdTree2dLInf}}

_lut_3d = {Metric.L1.value: {'f': _pt.KdTree3fL1,
                             'd': _pt.KdTree3dL1},
           Metric.L2Squared.value: {'f': _pt.KdTree3fL2Squared,
                                    'd': _pt.KdTree3dL2Squared},
           Metric.LInf.value: {'f': _pt.KdTree3fLInf,
                               'd': _pt.KdTree3dLInf}}

_lut_xd = {Metric.L1.value: {'f': _pt.KdTreeXfL1,
                             'd': _pt.KdTreeXdL1},
           Metric.L2Squared.value: {'f': _pt.KdTreeXfL2Squared,
                                    'd': _pt.KdTreeXdL2Squared},
           Metric.LInf.value: {'f': _pt.KdTreeXfLInf,
                               'd': _pt.KdTreeXdLInf}}


def _build_kd_tree(
        pts: npt.NDArray, metric: Metric, max_leaf_size: int, lut: dict):
    if pts.dtype != np.float32 and pts.dtype != np.float64:
        raise TypeError(
            "pts array format not supported: expected float32 or float64")

    return lut[metric.value][pts.dtype.char](pts, max_leaf_size)


def KdTree(pts: npt.NDArray, metric: Metric, max_leaf_size: int):
    """
    Create a KdTree. A KdTree is a binary tree that partitions space
    using hyper planes.

    Args:
        pts (NDArray): Input point set represented by a numpy NDArray of two
        dimensions.
        metric (Metric): Reference to a distance function.
        max_leaf_size (int): The maximum number of points contained by
            a leaf node.

    Returns:
        A KdTree instance.

    Raises:
        TypeError: Invalid numpy NDArray argument.
    """
    if pts.ndim != 2:
        raise TypeError("pts.ndim should equal 2")

    # All coordinates of a point have to be aligned. This in turn
    # allows us to determine the spatial dimension from either the row
    # or col count.
    dim = pts.shape[0] if pts.flags["F_CONTIGUOUS"] else pts.shape[1]

    if dim == 2:
        return _build_kd_tree(pts, metric, max_leaf_size, _lut_2d)
    elif dim == 3:
        return _build_kd_tree(pts, metric, max_leaf_size, _lut_3d)
    else:
        return _build_kd_tree(pts, metric, max_leaf_size, _lut_xd)
