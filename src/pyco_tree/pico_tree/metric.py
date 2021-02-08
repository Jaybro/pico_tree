from enum import Enum

__all__ = ['Metric']


class Metric(Enum):
    """
    The Metric class defines a set of names that can be used to refer to a specific distance function.

    Symbolic names:
        L1  Sum of absolute differences between point coordinates.
        L2  Sum of squared differences between point coordinates.
    """
    L1 = "L1"
    L2 = "L2"
