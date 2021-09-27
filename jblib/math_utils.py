# 3rd party
import numpy as np


def bilinear_interpolation_select(M, y, x):
    """ Selects the values from 2D matrix M given float coords y, x using
    bilinear interpolation.

    Args:
        M (np.array): (H, W)-matrix to select from.
        y (np.array of float32): (V, )-vector, y coordinates.
        x (np.array of float32): (V, )-vector, x coordinates.

    Returns:
        z (np.array): (V, )-vector of values selected from M. dtype = M.dtype.
    """

    y1 = np.floor(y)
    x1 = np.floor(x)

    p = y - y1
    q = x - x1

    y1 = y1.astype(np.int32)
    y2 = y1 + 1
    x1 = x1.astype(np.int32)
    x2 = x1 + 1

    z = (1.0 - p) * (1.0 - q) * M[y1, x1] + (1.0 - p) * q * M[y1, x2] + \
        p * (1.0 - q) * M[y2, x1] + p * q * M[y2, x2]

    return z


def deg2rad(a):
    """ Converts degrees to radians.

    Args:
        a (float or np.array): Scalar or D-dim array, D is arbitrary # of dims.

    Returns:
        float or np.array: Values converted to radians, same shape as a.
    """
    return a / 180.0 * np.pi


def rad2deg(a):
    """ Converts radians to degrees.

    Args:
        a (float or np.array): Scalar or D-dim array, D is arbitrary # of dims.

    Returns:
        float or np.array: Values converted to degrees, same shape as a.
    """
    return a / np.pi * 180.0
