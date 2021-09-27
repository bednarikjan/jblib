# Python std
import logging
from itertools import cycle

# 3rd party
import matplotlib.pyplot as plt
import numpy as np


def get_image_as_np_array(fig, closefig=True):
    """ Renders the figure `fig` and returns it as np.array.

    Args:
        fig (plt.Figrue): Figure.
        closefig (bool): The figure needs to be shown in order to draw into
            canvas. If False, the figure won't be closed afterwards (useful
            in case the figure was shown previously by other portion of
            code and this function should not close it as a side effect.)

    Returns:
        np.array of uint8: Image.
    """
    fig.show()
    fig.canvas.draw()
    if closefig:
        plt.close(fig)
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    return buf


def plot_tr_curves_lr(names, data, colors=None, line_styles=None,
                      line_widths=None, xlab='epochs', ylab_data='E',
                      ylab_lr='learning rate', logy_lr=False,
                      title='train/validation', legend=True, legend_loc='best',
                      grid=True, display=True, save=False, file=None,
                      xlim=None, ylim_data=None, ylim_lr=None, font_size=12):
    """ Creates and optionally displays and saves a plot with learning
    curves. The data andproperties of learning curves, e.g. names displayed in
    a legend, line styles/widths/colors, are given by `dict`s `names`,
    `data`, `colors`, `line_styles` and `line_widths` which map a name
    of the curve (a key in `data`) to the given property.

    Args:
        names (dict {str: str}: Mapping data name to displayed name.
        data (dict {str: list}): Mapping data name to list of values.
        colors (dict): {str: str}: E.g. {'err_tr': 'g'}
        line_styles (dict): {str: str}: E.g. {'err_tr': '--'}
        line_widths (dict {str: float}): E.g. {'err_tr': 1.0}
        xlab (str): x-axis kabel.
        ylab_data (str): y-axis flor data (left)
        ylab_lr (str): y-axis for learning rate (right)
        logy_lr (bool): Whether to display y-axis in log scale.
        title (str): Title.
        legend (bool): Whether to display legend.
        legend_loc (int or str): See `pytplot.legend`
        grid (bool): Show grid?
        display (bool): Show plot?
        save (bool): Save file?
        file (str): Absolute path to output file.
        xlim (tuple of float): Limits of x axis.
        ylim_data (tuple of float): Limits of data y axis.
        ylim_lr (tuple of float): Limits of lr y axis.
        font_size (int): Font size.
    """
    keys = sorted(list(names.keys()))

    if save and file is None:
        save = False
        logging.warning('File name not provided, the plot will not be saved.')

    # Check params colors, line styles, line widths.
    color_cycle = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    if colors is None:
        colors = {k: next(color_cycle) for k in keys}
    elif sorted(list(colors.keys())) != keys:
        logging.error('Different keys in "data" and "colors" provided.')
        print(list(colors.keys()))
        print(keys)
        return

    if line_styles is None:
        line_styles = {k: '-' for k in keys}
    elif sorted(list(line_styles.keys())) != keys:
        logging.error('Different keys in "data" "line styles" provided.')
        return

    if line_widths is None:
        line_widths = {k: 1.0 for k in keys}
    elif sorted(list(line_widths.keys())) != keys:
        logging.error('Different keys in "data" and "line widths" provided.')
        return

    # Create figure, axes.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot.
    lines = []
    for k in keys:
        d = data[k]
        n = names[k]
        c = colors[k]
        ls = line_styles[k]
        lw = line_widths[k]

        if n == 'lr':
            ln = ax2.plot(d, label=n, color=c, linestyle=ls, linewidth=lw)
        else:
            ln = ax1.plot(d, label=n, color=c, linestyle=ls, linewidth=lw)
        lines.extend(ln)

    # Set axes labels, ranges, scales.
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab_data)
    ax2.set_ylabel(ylab_lr)

    # Set axes limits.
    if ylim_data is not None:
        if not isinstance(ylim_data, tuple) and not isinstance(ylim_data, list):
            ylim_data = [0.0, ylim_data]
        ax1.set_ylim(ylim_data)
    if xlim is not None:
        if not isinstance(xlim, tuple) and not isinstance(xlim, list):
            xlim = [0.0, xlim]
        ax1.set_xlim(xlim)
    if ylim_lr is not None:
        if not isinstance(ylim_lr, tuple) and not isinstance(ylim_lr, list):
            ylim_lr = [1e-8, ylim_lr]
        ax2.set_ylim(ylim_lr)

    if logy_lr:
        ax2.set_yscale('log')
    plt.title(title)
    if legend:
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=legend_loc)
    if grid:
        ax1.grid()
    plt.rc('font', size=font_size)  # controls default text sizes

    # Supress margins (pad value is a fraction of the font-size)
    plt.tight_layout(pad=1.0)

    if save:
        plt.savefig(file, dpi=200)
    if display:
        plt.show()


def vis_rect_mesh_func(xy, v, cmap='jet', shading='gouraud', flip_y=True,
                       vrange='auto', vrange_margin=0.1,
                       return_img=True, show=False):
    """

    Args:
        xy (np.array): xy coordinates of vertices, shape (H, W, 2)
        v (np.array): values corresponding to vertices, shape (H, W)
        flip_y (bool): plt.pcolormesh assumes (0, 0) as lower left corner with
            y-axis going up, whereas most often the coordinate system is
            with y-axis going down. If True, the mesh `xy` is rotated by 180
            deg around x-axis.
        vrange (None or str or tuple): Range of values used for encoding color.
            If None, min and max of `v` are used. If `auto`, min and max values
            of `c` after leaving out biggest and lowest `vrange_margin`
            fraction of sorted `c` are used. If tuple, then first value is min,
            second is max.
        vrange_margin (float): See `vrange`. In range [0, 1].
        return_img (bool): Whether to return the image as np array.
        show (bool): Whether to display the plot.

    Returns:
        None or np.array: If `return_img`, returns the image content
            of the plot.
    """

    if isinstance(vrange, str):
        if vrange in {'auto', 'auto_center_zero'}:
            vrange_margin = np.maximum(0.0, np.minimum(1.0, vrange_margin))
            cs = np.sort(v.flatten())
            num_vals = cs.shape[0]
            mrg = int(vrange_margin * num_vals)
            vmin = cs[mrg]
            vmax = np.maximum(vmin, cs[-mrg])

            if vrange == 'auto_center_zero':
                vlim = np.maximum(np.abs(vmin), np.abs(vmax))
                vmin = -vlim
                vmax = vlim
        else:
            raise Exception('Unknown vrange mode "{}"'.format(vrange))

    elif isinstance(vrange, tuple) or isinstance(vrange, list):
        vmin, vmax = vrange
    else:
        vmin = None
        vmax = None

    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(xy[:, :, 0], xy[:, :, 1] * (1.0, -1.0)[flip_y], v,
                  cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    if show:
        plt.show()

    if return_img:
        return get_image_as_np_array(fig, closefig=(not show))



### Tests
if __name__ == "__main__":
    ### Test vis_rect_mesh_func
    x, y = np.meshgrid(np.arange(3), np.arange(3))
    xy = np.stack([x, y], axis=2)
    # z = np.random.uniform(0, 1, (3, 3))
    z = np.arange(9).reshape((3, 3))

    img = vis_rect_mesh_func(xy, z, flip_y=False, show=False, return_img=True)
