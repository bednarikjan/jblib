# Python std.
import logging

# 3rd party
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# 3dsr
import jblib.file_sys as jbfs


img_dtypes = {
    'float32': np.float32,
    'uint8': np.uint8
}


def normalize(img, mode='global'):
    """ Normalizes the image by subtracting the mean and dividing by std.
    Computation of mean and std and what pixels they get applied to can
    be changed using `mode`.

    Args:
        img (np.array of float32): RGB (H, W, 3) or GS (H, W) image.
        mode (str): Normalization mode, one of:
            * 'global': Mean and std over the whole image.
            * 'channels': Mean and std channel-wise.

    Returns:
        np.array of float32: Normalized image.
    """
    if img.dtype is not np.dtype(np.float32):
        raise Exception('Input image must have type float32.')

    if img.ndim == 2 and mode == 'channel':
        mode = 'global'

    if mode == 'global':
        imgn = (img - np.mean(img)) / np.std(img)
    elif mode == 'channel':
        imgn = (img - np.mean(img, keepdims=True)) / np.std(img, keepdims=True)
    elif mode == 'imagenet':
        imgn = img - np.array([103.939, 116.779, 123.68]) / 255.0
    else:
        raise Exception('Unknown normalization mode "{}".'.format(mode))

    return imgn


def load(path, dtype='float32', keep_alpha=False):
    """ Loads the image and converts it to one of following given the `dtype`:
    uint8   - pixels in [0, 255]
    float32 - pixels in [0.0, 1.0]

    Args:
        path (str): Absolute path to file.
        dtype (str): Output data type with corresponding value range.
            One of {'uint8', 'float32'}
        keep_alpha (bool): Whether to keep alpha channel. Only applies for
            RGB images and alpha is assumed to be 4th channel.

    Returns:
        np.array: RGB or GS image, possibly with 4th alpha channel (if
        the input image has it and `keep_alpha is True`). Data type and pixel
        values range is given by `dtype`.
    """
    img = plt.imread(path)

    # Check dtype of the image.
    if img.dtype not in (np.uint8, np.float32):
        raise Exception('Loaded image {p} has unsupported dtype {dt}. '
                        'load_cvt() only supports one of (uint8, float32).'.
                        format(p=path, dt=img.dtype))

    # Check dtype argument.
    if dtype not in ('uint8', 'float32'):
        raise Exception('Supported values for "dtype" are ("uint8, float32"), '
                        'got {dt}'.format(dt=dtype))

    # Keep or remove alpha channel.
    if img.ndim == 3 and img.shape[2] == 4 and not keep_alpha:
        img = img[..., :3]

    # Convert data type.
    if img.dtype == np.uint8 and dtype == 'float32':
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.float32 and dtype == 'uint8':
        img = np.round(img * 255.0).astype(np.uint8)

    return img


def load_n(paths, dtype='float32', keep_alpha=False, norm=True,
           norm_mode='global'):
    """ Loads images each given by path in `paths`.

    Args:
        paths (str or list of str): Absolute paths to images.
        dtype (str): Output data type, one of {'float32', 'uint8'}.
        keep_alpha (bool): Whether to keep alpha channel.
        norm (bool): Whether to normalize the images.
        norm_mode (str): Normalization mode.

    Returns:
        np.array: (N, H, W, 3) or (N, H, W) tensor for RGB or GS images.
    """
    if isinstance(paths, str):
        paths = [paths]

    imgaux = plt.imread(paths[0])
    n_files = len(paths)

    sh = imgaux.shape
    if len(sh) == 3 and sh[2] == 4 and not keep_alpha:
        sh[2] = 3

    images = np.empty((n_files, ) + sh, dtype=img_dtypes[dtype])

    for i in range(n_files):
        img = load(paths[i], dtype=dtype, keep_alpha=keep_alpha)
        if norm:
            img = normalize(img, norm_mode)
        images[i] = img

    return images


def load_dirs(paths, exts=None, dtype='float32', keep_alpha=False,
              norm=True, norm_mode='global', with_names=False,
              verbose=False):
    """ Loads images contained in one or more directories given in `path`
    and returns them. Function expects that all images are of the same shape
    (width, height, channels).

    Args:
        paths (str or list of str): Absolute path(s) to directory(ies).
        exts (str or list of str or None): File extension. If `None`, all
            files are loaded.
        dtype (str): Output data type, one of {'float32', 'uint8'}.
        keep_alpha (bool): Whether to keep alpha channel.
        norm (bool): Whether to normalize the images.
        norm_mode (str): Normalization mode.
        with_names (bool): If true, list of list of file names is returned
            as well.

    Returns:
        np.array: (N, H, W, 3) or (N, H, W) tensor for RGB or GS images.
        list of list of str: File names. Only if `with_names`.
    """
    if isinstance(paths, str):
        paths = [paths]

    files_in_dirs = []
    n_files = 0
    for p in paths:
        files = jbfs.ls(p, exts)
        nf = len(files)
        if nf == 0:
            logging.warning('No files with extensions "{}" found in '
                            'directory {}'.format(exts, p))
        n_files += nf
        files_in_dirs.append(files)

    if n_files == 0:
        raise Exception('No files with extensions "{}" found in dirs "{}"'.
                        format(exts, paths))

    # Load one image and find out its shape.
    imgaux = None
    for p, fs in zip(paths, files_in_dirs):
        if len(fs) > 0:
            imgaux = plt.imread(jbfs.jn(p, fs[0]))
            break

    # Find shape according to whether we keep alpha.
    sh = imgaux.shape
    if len(sh) == 3 and sh[2] == 4 and not keep_alpha:
        sh[2] = 3

    # Load all images.
    num_dirs = len(paths)
    images = np.empty((n_files, ) + sh, dtype=img_dtypes[dtype])
    i = 0
    for pi, (p, files) in enumerate(zip(paths, files_in_dirs)):
        num_files_in_dir = len(files)
        for fi, f in enumerate(files):
            if verbose:
                print('\rLoading dir {}/{}, file {}/{}'.
                      format(pi + 1, num_dirs, fi + 1, num_files_in_dir),
                      end='')

            img = load(jbfs.jn(p, f), dtype=dtype, keep_alpha=keep_alpha)
            if norm:
                img = normalize(img, norm_mode)
            images[i] = img
            i += 1
    if with_names:
        return images, files_in_dirs
    else:
        return images


def save(path, img):
    """ Stores the image to the disk. Number of channels is preserved and
    the format is inferred from the extension in the `path`.

    Note:
        Not using `pyplot.imsave()`, since it forces storing alpha channel.

    Args:
        path (str): Absolute path.
        img (np.array): Image data.
    """
    # Check image data type.
    if img.dtype not in (np.uint8, np.float32):
        raise Exception('Input image must have one of the following data '
                        'types: (np.uint8, np.float32), got {dt}.'.
                        format(dt=img.dtype))

    # Convert to uint8 (to be able to construct PIL.Image).
    if img.dtype == np.float32:
        img = np.round(img * 255.0).astype(np.uint8)

    # Convert to PIL.Image and save.
    Image.fromarray(img).save(path)


def resize(img, w_new, interpolation='nn'):
    """ Resizes the image so that its new width would be `newWidth` and the
    aspect ratio would be preserved.

    Args:
        img (np.array): (H, W) or (H, W, 3)-tensor. Input image.
        w_new (int): New width.
        interpolation (str): Interpolation type.

    Returns:

    """
    if interpolation == 'nn':
        intp = cv2.INTER_NEAREST
    elif interpolation == 'bilin':
        intp = cv2.INTER_LINEAR
    else:
        raise Exception('Not supported interpolation type "{}".'.
                        format(interpolation))

    if img.ndim < 2 or img.ndim > 3:
        raise Exception('Input image must have 2 or 3 dimensions, found {}'.
                        format(img.ndim))

    sc = float(w_new) / img.shape[1]
    new_height = np.round(sc * img.shape[0]).astype(np.int32)

    img_res = cv2.resize(img, (w_new, new_height), interpolation=intp)
    return img_res

def get_mask(img, fgrd=True):
    """ Returns the binary mask where foregournd corresponds to non-zero
    pixels (at least one channel has to be non-zero). `fgrd` flag controls
    whether foreground pixels are True or False.

    Args:
        img (np.array of float): (H, W, 3)-tensor, (H, W)-image of 3D normals.
        fgrd (bool): Whether to return foreground mask.

    Returns:
        np.array of bool: (H, W)-matrix, True - foreground, False - background.
    """

    mask = np.sum(np.abs(img), axis=2).astype(np.bool)

    if not fgrd:
        mask = (1 - mask).astype(np.bool)

    return mask

def get_mask_batch(imgs, fgrd=True):
    """ Returns the binary masks where foregournd corresponds to non-zero
    pixels (at least one channel has to be non-zero). `fgrd` flag controls
    whether foreground pixels are True or False.

    Args:
        imgs (np.array of float): (B, H, W, 3)-tensor,
            B is batch size, (H, W)-image of 3D normals.
        fgrd (bool): Whether to return foreground mask.

    Returns:
        np.array of bool: (B, H, W)-matrix, True - foreground,
            False - background.
    """

    mask = np.sum(np.abs(imgs), axis=3).astype(np.bool)

    if not fgrd:
        mask = (1 - mask).astype(np.bool)

    return mask


def draw_points(img, coords, color=(0, 0, 255), radius=1, thickness=1, overwrite=True):
    """ Draws the points into image `img` given `coords`.

    Args:
        img (np.array of uint8 or float32):  (H x W x Ch)-tensor, Ch-channeled
            image. For color, expected order BGR. If dtype is float, then the
            image is likely in range [0.0, 1.0] and `color` thus needs
            to be set as float within range [0.0, 1.0] as well.
        coords (np.array int32): (N x 2)-matrix, N is # points. Each row is
            [x, y] coordinate. If float values are passed, they are rounded and
            casted to int32.
        color (tuple of int32 or float): Color of points, (B, G, R), range
            [0, 255] or range [0.0, 1.0].
        radius (int): Point radius in px.
        thickness (int): Point edge thickness in px.
        overwrite (bool): Whether to draw directly in the `img` or rather
            create a copy.

    Returns:
    img (np.array): Image where points have been drawn.
    """

    if coords.dtype != np.int32:
        coords = np.round(coords).astype(np.int32)

    img_draw = img if overwrite else np.copy(img)

    for c in coords:
        cv2.circle(img_draw, (c[0], c[1]), radius, color, thickness=thickness)

    return img_draw
