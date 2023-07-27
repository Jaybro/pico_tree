from enum import Enum

__all__ = ['Metric']


class Metric(Enum):
    """
    The Metric class defines a set of names that can be used to refer
    to a specific distance function.

    Symbols:
        L1: Sum of the absolute differences between point coordinates.
        L2Squared: Sum of the squared differences between point
            coordinates.
        LInf: Max of the absolute differences between point
            coordinates.
    """
    L1 = "L1"
    L2Squared = "L2Squared"
    LInf = "LInf"
