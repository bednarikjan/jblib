# 3rd party
import numpy as np
import cv2
from scipy import sparse
from scipy.sparse import linalg as spla

# project files
from . import img as jbim


# Supported coordinate frames.
COORD_FRAMES = \
    {'ocv',  # x right, -y up,  z forward
     'ogl'   # x right,  y up, -z forward
     }

# Rotation Tfs from default frame (OGL) to other coordinate frames. E.g. given
# a point p_ogl in default frame (OGL), we would get it's repr. in OCV
# frame by p_ocv = TFS['ocv'] @ p_ogl.
TFS = {
    'ogl': np.identity(3, np.float32),
    'ocv': np.array([[1.,  0.,  0.],
                     [0., -1.,  0.],
                     [0.,  0., -1.]])
}


def check_coord_frames(frames):
    """ Checks whether `frame` is supported and raises exception otherwise.

    Args:
        frames (str or list of str): Frame(s) identifier(s).
    """

    if isinstance(frames, str):
        frames = [frames]
    if not isinstance(frames, list):
        raise Exception('Arg "frame" has to be str or list.')

    for f in frames:
        if f not in COORD_FRAMES:
            raise Exception('Unsupported coordinate frame "{}". '
                            'Must be one of {}'.format(frames, COORD_FRAMES))


def normals2img(n, mask_fgrd=None, frame='ocv'):
    """ Converts the predicted normals tensor to the image representation.
    Normals 'n' must be given in one of the following coordinate `frame`:
        ocv: x right, y down, z forward
        ogl: x right, y up, -z forward
    Before conversion, the normals are transformed into ogl frame. Then
    the conversion is as follows:

    x [-1.0, 0.0] -> [0,   255]
    y [-1.0, 1.0] -> [0,   255]
    z [0.0,  1.0] -> [128, 255]

    Args:
        n (np.array of float32): (H, W, 3)-tensor, H x W matrix of 3D normal
            vectors.
        mask_fgrd (np.array): (H, W)-matrix, binary {0, 1}, specifies the mask
            of the foreground (object). If it is given, the resulting image is
            multiplied with it to suppress whatever normals are in the
            backrgound part. If None, the mask is inferred from `n` by taking
            only those (y, x) locations where sum(n[y, x], axis=2) = 0, i.e.
            black backrgound.

    Returns:
        np.array of uint8: (H, W, 3)-tensor. Image representation of normals.
    """

    check_coord_frames(frame)

    # TF from frame back to default coord. frame (OGL).
    M = TFS[frame].T

    nstd = np.transpose(M @ np.transpose(n, (0, 2, 1)), (0, 2, 1))
    nimg = np.clip(np.round(0.5 * (nstd + 1.0) * 255.0), 0.0, 255.0).\
        astype(np.uint8)

    if mask_fgrd is None:
        mask_fgrd = np.mean(np.abs(n), axis=2).astype(np.bool).astype(np.int32)

    return (nimg * mask_fgrd[..., None]).astype(np.uint8)


def normals2depth(nmap, s=1.0, t=0.0):
    """ Computes depth map from normal map using least squares and finite
    differences in depth [1]. Orthographic camera model is assumed, however,
    this normally works pretty well even if the data come from perspective
    (pinhole) camera model. The actual normals do not need to be rectangular,
    but the background is required to be [0, 0, 0].

    This function is counterpart to utils.depth_utils.depth2normals()

    Expected coordinate frame as in OpenCV:

      _ z
      /|
     /
    +---> x (= u)
    |
    |
    v y (= v)

    [1] Fua. P - Computer Vision course, slides ShapeFromX

    Args:
        nmap (np.array): (H, W, 3)-tensor, (H, W)-image of unit-length 3D
            normals.
        s (float): Scale mapping real size to pixel size. s = dx/du = dy/dv,
            i.e. assuming orthographic projection, s tells how big one pixel is
            in real unit (e.g. meters). Since the reconstruction is up to scale
            s, if set properly, the reconstruction would best relate to real
            object.
        t (float): Translation. The reconstruction is up to translation.
            It seems that the reconstruction tends to be centered around 0.

    Returns:
        dmap (np.array): (H, W)-array of pixel-wise depths (depth corresponds
            to Z axis).
    """

    def extend_mask_rd(mask, iters=1):
        """ Extends the mask (foreground) one step to right and down. Example:

        0 0 0 0 0      0 0 0 0 0
        0 1 1 0 0      0 1 1 1 0
        0 1 0 0 0  ->  0 1 1 0 0
        0 1 1 1 1      0 1 1 1 1
        0 0 1 0 0      0 0 1 1 0

        Args:
            mask (np.array): (H, W)-matrix, binary array with 1 = foreground,
                0 = background
            iters (int): Number of iterations to extend the mask.

        Returns:
            np.array: (H, W)-matrix of the same type as `mask`.
        """

        kernel = np.array([[1, 1], [1, 0]], dtype=np.uint8)

        # We are doing convolution, thus kernel needs to be flipped vertically and horizontally.
        kernel = cv2.flip(cv2.flip(kernel, 0), 1)

        m = mask.astype(np.uint8)
        m = cv2.dilate(m, kernel, iterations=iters)

        return m.astype(mask.dtype)

    # Extending the normals map by 2 pixels along right and bottom edge.
    nme = np.zeros((nmap.shape[0] + 2, nmap.shape[1] + 2, 3), dtype=nmap.dtype)
    nme[:-2, :-2] = nmap

    # Copy the normals on step to the right and down - to prevent ill-posed leas-squares problem.
    m1 = jbim.get_mask(nme, fgrd=False)
    nme[:, 1:] += nme[:, :-1] * m1[:, 1:][..., None]
    m2 = jbim.get_mask(nme, fgrd=False)
    nme[1:, :] += nme[:-1, :] * m2[1:, :][..., None]

    # Mask of non-zero normals.
    mask_n = jbim.get_mask(nme)
    # Mask of non-zero depth values (to be reconstructed).
    mask_z = extend_mask_rd(mask_n)

    # Get dimensions of the matrix A for the LS system Ax = b
    Arows = np.sum(mask_n) * 2
    Acols = np.sum(mask_z)

    ### Create sparse matrix A.
    inds_z = np.copy(mask_z).astype(np.int32)
    inds_z[inds_z == 0] = -1
    inds_z[inds_z != -1] = np.arange(Acols)

    # minus ones
    inds_z_tl = inds_z[mask_n].flatten()

    i_m_ones = np.arange(Arows)
    j_m_ones = np.stack([inds_z_tl] * 2, axis=1).flatten()
    data_m_ones = -np.ones((Arows,))

    # ones from right
    mask_ones_r = np.zeros_like(mask_n, dtype=np.bool)
    mask_ones_r[:, 1:] = mask_n[:, :-1]
    inds_z_r = inds_z[mask_ones_r].flatten()

    i_ones_r = np.arange(Arows // 2) * 2
    j_ones_r = inds_z_r
    data_ones_r = np.ones((Arows // 2,))

    # ones from down
    mask_ones_d = np.zeros_like(mask_n, dtype=np.bool)
    mask_ones_d[1:, :] = mask_n[:-1, :]
    inds_z_d = inds_z[mask_ones_d].flatten()

    i_ones_d = i_ones_r + 1
    j_ones_d = inds_z_d
    data_ones_d = np.ones((Arows // 2,))

    # Concat to get indices and data for creating sprase matrix.
    data = np.concatenate((data_m_ones, data_ones_r, data_ones_d), axis=0)
    i_inds = np.concatenate((i_m_ones, i_ones_r, i_ones_d), axis=0)
    j_inds = np.concatenate((j_m_ones, j_ones_r, j_ones_d), axis=0)

    As = sparse.coo_matrix((data, (i_inds, j_inds)), (Arows, Acols))

    ### Build dense vector b of the LS system Ax = b
    b = -(nme[mask_n][:, :2] / nme[mask_n][:, 2][:, None]).flatten()

    # Solve for unknown x: Ax = b.
    z = spla.lsmr(As, b)[0]

    # Scale and translate z.
    z = z * s + t

    # Create depth map.
    dmap = np.zeros_like(mask_z, dtype=np.float64)
    dmap[mask_z] = z

    # Delete artificially constructed depth values on the very right and bottom.
    dmap[jbim.get_mask(nme, fgrd=False)] = 0.0

    # Get the depth map of the original normal map size.
    dmap = dmap[:-2, :-2]

    return dmap


def transform(nmap, frame_from, frame_to):
    """ Transforms the normals from one coordinate frame to another,
    the transformatio is assumed to be rotation only!

    Args:
        nmap (np.array): Normal map (H, W, 3).
        frame_from (str): Original coordinate frame.
        frame_to (str): New coordinate frame.

    Returns:
        np.array: Transformed normals, (H, W, 3).
    """

    check_coord_frames([frame_from, frame_to])
    M = TFS[frame_to] @ TFS[frame_from].T

    return (M @ nmap.transpose((0, 2, 1))).transpose((0, 2, 1))


# Tests.
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    cwd = os.path.dirname(__file__)

    ############################################################################
    # TEST transform().
    print('Testing transform()')
    print('===================')

    path_data = os.path.join(cwd, 'tests/data_test_normals')
    path_nmap_ocv = os.path.join(path_data, 'n_ocv.npz')
    path_nmap_ogl = os.path.join(path_data, 'n_ogl.npz')

    n_ocv = np.load(path_nmap_ocv)['normals']
    n_ogl = np.load(path_nmap_ogl)['normals']

    n_ocv_2 = transform(transform(n_ocv, 'ocv', 'ogl'), 'ogl', 'ocv')
    assert(np.allclose(n_ocv, n_ocv_2))

    n_ocv_3 = transform(n_ocv, 'ocv', 'ocv')
    assert (np.allclose(n_ocv, n_ocv_3))

    # The following images need to be visually inspected and they need to look
    # like our ordinary nmaps.
    n_ogl_from_ocv = transform(n_ocv, 'ocv', 'ogl')
    n_ogl_from_ocv_im = normals2img(n_ogl_from_ocv, frame='ogl')
    plt.imsave(os.path.join(path_data, 'n_ogl.png'), n_ogl_from_ocv_im)

    n_ocv_from_ogl = transform(n_ogl, 'ogl', 'ocv')
    n_ocv_from_ogl_im = normals2img(n_ocv_from_ogl, frame='ocv')
    plt.imsave(os.path.join(path_data, 'n_ocv.png'), n_ocv_from_ogl_im)
    print('[OK]')
