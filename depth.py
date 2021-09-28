# 3rd party
import numpy as np

# project files
from . import math_utils as jbmath


def get_mask(dmaps, fgrd=True):
    """ Returns the binary mask where foregournd corresponds to non-zero depth.
    `fgrd` flag controls whether foreground pixels are True or False.

    Args:
        dmaps (np.array of float): (H, W) or (N, H, W)-tensor, N is batch size,
            (H, W)-shaped depth map.
        fgrd (bool): Whether to return foreground or background mask. If True,
            foreground pixels are True, otherwise False.

    Returns:
        masks (np.array of bool): (H, W) or (N, H, W)-tensor, True - foreground,
            False - background (unless `fgrd` == False).

    """

    if dmaps.ndim != 2 and dmaps.ndim != 3:
        raise Exception('Depth maps "dmaps" must have 2 or 3 dimensions of '
                        'shape (H, W) or (N, H, W), {} dims found.'.
                        format(dmaps.ndim))

    masks = dmaps.astype(np.bool)

    if not fgrd:
        masks = (1 - masks).astype(np.bool)

    return masks


def dmap2pcloud(dmap, K, yx=None, interpolation='nn',
                num_points=None, skip_zero_depth=True):
    """ Generates the point cloud from given depth map `dmap` using intrinsic
    camera matrix `K`. It can operate in 2 modes. Mode 1 is selected, if
    `coords` is not None, mode 2 is selected otherwise.

    Modes:
    1) User provides the 2d yx coordinates `coords` to be backprojected,
    2) User specifices the number of points `numPoints` to be selected on
    random from `dmap` and backprojected, or if numPoints='all' is specified,
    then all non-zero depth points are used.

    Args:
        dmap (np.array of float32): (H x W)-matrix, 1 channeled image.
        K (np.array of float32): (3 x 3)-matrix, camera intrinsics.
        yx (np.array of float32): (V x 2)-matrix, V is # of 2D [y, x]
            coordinates. If None, mode 2 is used.
        interpolation (str): Type of interpolation when selecting value from
            depth map given float [y, x] coordinates. Allowed values are
            {'nn', 'bilinear'}. 'nn' - nearest neighbour (i.e. no
            interpolation), 'bilinear' - bilinear interpolation (value selected
            from 2x2 patch).
        num_points (int or str): Number of points to be backprojected. Only
            applicable in mode 2.
        skip_zero_depth (bool): Only applicable in mode 2. It is expected that
            the depth map contains "black" parts, i.e. zero values, which are,
            however, not plausible. If `skipZeroDepth` is set ti True, only
            nonzero depth map pixels will be considered for backprojection.

    Returns:
        pcloud (np.array of float32): (N x 3)-matrix, N is `coords.shape[0]`
            in mode 1 or `numPoints` in mode 2.
    """

    if dmap.ndim != 2:
        raise Exception('Depth map must be 1 channeled image, i.e. shape must '
                        'be (H, W). Found {} dims.'.format(dmap.ndim))

    # Get inverted instrinsic matrix.
    Kinv = np.linalg.inv(K)

    # Mode 1 - provided coords.
    if yx is not None:
        N = yx.shape[0]

        if interpolation == 'nn':
            yx = np.round(yx).astype(np.int32)
            x = yx[:, 1]
            y = yx[:, 0]
            z = dmap[y, x]
        elif interpolation == 'bilinear':
            x = yx[:, 1]
            y = yx[:, 0]
            z = jbmath.bilinear_interpolation_select(dmap, y, x)
        else:
            raise Exception('Unknown interpolation type "{}".'.
                            format(interpolation))

    # Mode 2 - random coords.
    else:
        if isinstance(num_points, str) and num_points == 'all':
            y, x = np.where(dmap != 0.0)
            N = y.shape[0]
        elif isinstance(num_points, int):
            N = num_points

            if skip_zero_depth:
                dm_inds = np.where(dmap)
            else:
                H, W = dmap.shape
                mg = np.meshgrid(np.arange(W), np.arange(H))
                dm_inds = (mg[1].flatten(), mg[0].flatten())

            N = np.minimum(N, dm_inds[0].shape[0])

            pt_rand_inds = np.sort(np.random.permutation(len(dm_inds[0]))[:N])
            x = dm_inds[1][pt_rand_inds]
            y = dm_inds[0][pt_rand_inds]
        else:
            raise Exception('Parameter "num_points" must have type str or int,'
                            'got {}'.format(type(num_points)))

        z = dmap[y, x]

    pts_proj = np.vstack((x[None, :], y[None, :], np.ones((1, N))) * z[None, :])
    pcloud = (Kinv @ pts_proj).T

    return pcloud.astype(np.float32)


def dmap2img(dmap, mode='auto', range=None):
    """ Converts the depth map to an RGB image for visualization purposes.
    Depth map stores the values corresponding to physical depth. This functions
    transforms the values to standard image range [0, 255]. It tries to equalize
    the histogram of depths, i.e. it assumes gaussian distribution of depths
    within an image and it finds the minimum and maximum depth by
    subtracting/adding 3 times the std to the mean.

    Args
    dmap (np.array of float32): Depth map, shape (H, W).
    mode (str): Mode to choose the range of depth values to display.
        Allowed values:
            'auto':
            'minmax_scaled':
            'custom':
    range (tuple):

    Returns:
        img (np.array of uint8): (H, W, 3)-tensor, image.
    """

    # Convert to floating point type.
    dmap = dmap.astype(np.float32)

    if mode == 'auto':
        # Find statistics.
        mu = np.mean(dmap[dmap != 0])
        std = np.std(dmap[dmap != 0])

        # Find limts.
        d_max = mu + 3 * std
        d_min = mu - 3 * std

        # Normalize.
        dm_img = (dmap - d_min) / (d_max - d_min)

        # Clip.
        dm_img[dm_img < 0.0] = 0.0
        dm_img[dm_img > 1.0] = 0.0

    elif mode == 'minmax_scaled':
        d_min = np.min(dmap[dmap != 0])
        d_max = np.max(dmap[dmap != 0])

        dm_img = np.copy(dmap)
        dm_img[dm_img != 0.0] = ((dm_img[dm_img != 0.0] - d_min) /
                                 (d_max - d_min)) * 0.8 + 0.1
    elif mode == 'custom':
        d_min, d_max = range
        dm_img = np.copy(np.clip(dmap, d_min, d_max))
        dm_img[dm_img != 0.0] = ((dm_img[dm_img != 0.0] - d_min) /
                                 (d_max - d_min))
    else:
        raise Exception('Unknown mode "{}"'.format(mode))

    # Convert to 3-chaneled uint8 image.
    dm_img = (dm_img * 255.0).astype(np.uint8)
    dm_img = np.stack([dm_img] * 3, axis=2)

    return dm_img


def create_topology(dmap):
    """ Finds the faces connecting neighboring non-zero depth values. Each face
    is a triangle consisting of three integers pointing to an array of indices
    of non-zero cells. Non-zero cells are ordered along x-axis (first)
    and y-axis (second).

    Args:
        dmap (np.array of float32): Dmap, shape (H, W).

    Returns:
        np.array of int32: Faces, shape (F, 3)
    """
    assert dmap.ndim == 2

    # Zero-pad the dmap.
    dmap = np.pad(dmap, 1, 'constant', constant_values=(0., ))  # (H + 1, W + 1)

    # Extract coordinates of non-zero cells.
    cy, cx = np.stack(np.nonzero(dmap), axis=1).T  # each (N, )

    # Get a 2D map of indices of non-zero values
    imap = dmap.copy().astype(np.bool).astype(np.int32)
    imap[imap != 0] = np.arange(1, np.sum(imap) + 1)

    # Get faces to the bottom.
    fbot = np.stack([imap[cy, cx], imap[cy + 1, cx],
                     imap[cy + 1, cx + 1]], axis=1)
    fbot = fbot[np.min(fbot, axis=1) != 0]  # (FB, 3)

    # Get faces to the right.
    fright = np.stack([imap[cy, cx], imap[cy + 1, cx + 1],
                       imap[cy, cx + 1]], axis=1)
    fright = fright[np.min(fright, axis=1) != 0]  # (FR, 3)

    # Get faces to the right reversed
    frightr = np.stack([imap[cy, cx], imap[cy + 1, cx],
                        imap[cy, cx + 1]], axis=1)
    frightr = frightr[(np.min(frightr, axis=1) != 0) &
                      (imap[cy + 1, cx + 1] == 0)]  # (FRR, 3)

    # Get faces to the left.
    fleft = np.stack([imap[cy, cx], imap[cy + 1, cx - 1],
                      imap[cy + 1, cx]], axis=1)
    fleft = fleft[(np.min(fleft, axis=1) != 0) &
                  (imap[cy, cx - 1] == 0)]  # (FL, 3)

    # Get all faces.
    return np.concatenate([fbot, fright, frightr, fleft], axis=0) - 1

