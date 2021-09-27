""" Linear algebra utils.
"""

# 3rd party
import numpy as np


def normalize(v):
    """ Normalizes the input vector `v` such that it becomes unit length.

    Args:
        v (np.array): (D, )-vector.

    Returns:
        np.arrray: (D, )-vector, unit length.
    """
    if v.ndim != 1:
        raise Exception('Input vector must have 1 dimension, {} found.'.
                        format(v.ndim))

    return v / np.linalg.norm(v)
