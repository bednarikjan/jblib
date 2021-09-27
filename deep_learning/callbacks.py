# Python std.
import os
import abc
from abc import abstractmethod

# 3dsr
import jblib.file_sys as jbfs
import jblib.vis2d as jbv2
import jblib.normals as jbn
import jblib.img as jbim

# 3rd party
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Callback(abc.ABC):
    def __init__(self, period):
        self.set_period(period)

    @abstractmethod
    def on_epoch_end(self, epoch, force_run=False, **kwargs):
        raise NotImplementedError

    def get_period(self):
        return self._period

    def set_period(self, period):
        if not isinstance(period, int) or period < 1:
            raise Exception('"period" has to be integer number greater than 0.')
        self._period = period

    @abstractmethod
    def get_file_name(self):
        raise NotImplementedError

    @abstractmethod
    def set_file_name(self, fname):
        raise NotImplementedError

    @abstractmethod
    def add_file_name_suffix(self, suff):
        """ Adds suffix to the current file name. In case
        of file (not directory), it adds it before the extension.

        Args:
            suff (str): Suffix to add.
        """
        raise NotImplementedError


class HistorySaver(Callback):
    """ Callback for svaing the training history to .npz file.
    """

    def __init__(self, file_name, period=10):
        """
        Args:
            file_name (str): Absolute path to output history file.
            period (int): Interval between saving current history in epochs.
        """
        if not os.path.exists(os.path.dirname(file_name)):
            raise Exception('The directory "{d}" intended for saving '
                            'the learning history does not exist.'.
                            format(d=os.path.dirname(file_name)))

        super(HistorySaver, self).__init__(period)

        self._fname_base = jbfs.unique_file_name(file_name)
        self._curr_fname = None

    def on_epoch_end(self, ep, force_run=False, **kwargs):
        if ep % self._period == 0 or force_run:
            history = kwargs['history']

            # Store last file name.
            old_file_name = self._curr_fname

            # Generate new file name.
            self._curr_fname = self._fname_base[:-4] + \
                               '_epoch_{:03d}'.format(ep) + '.npz'
            self._curr_fname = jbfs.unique_file_name(self._curr_fname)

            # Save.
            np.savez_compressed(self._curr_fname, **history)

            # Remove old file.
            if old_file_name is not None:
                os.remove(old_file_name)

    def get_file_name(self):
        return self._fname_base

    def set_file_name(self, fn):
        self._fname_base = fn
        self._curr_fname = None

    def add_file_name_suffix(self, suff):
        fn = self.get_file_name()
        nam, ext = jbfs.split_name_ext(fn)
        self.set_file_name(nam + suff + '.' + ext)


class WeightsSaverTF(Callback):
    """ Callback for saving the current weights each `period` epochs. TF's
    `tf.Train.Saver` is used for this purpose.
    """
    def __init__(self, file_name, sess,
                 period=10, keep_chkpt_every_n_hours=10000.0,
                 max_to_keep=1):
        """
        Args:
            file_name (str): Absolute path to checkpoint file.
            sess (tf.Session): TF session.
            period (int): Interval between saves, in epochs.
            keep_chkpt_every_n_hours (float): See `tf.train.Saver`.
            max_to_keep (int): See `tf.train.Saver`.
        """
        super(WeightsSaverTF, self).__init__(period)

        self._fname = file_name
        self._sess = sess
        self._saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=keep_chkpt_every_n_hours,
            max_to_keep=max_to_keep)

    def on_epoch_end(self, ep, force_run=False, **kwargs):
        if ep % self._period == 0 or force_run:
            self._saver.save(self._sess, self._fname, global_step=ep)

    def get_file_name(self):
        return self._fname

    def set_file_name(self, fn):
        self._fname = fn

    def add_file_name_suffix(self, suff):
        fn = self.get_file_name()
        nam, ext = jbfs.split_name_ext(fn)
        self.set_file_name(nam + suff + '.' + ext)


class PlotsSaver(Callback):
    """ Saves the plots of given learning curves. It only saves the plots once
    in `period` epochs. It saves 3 same plots but with different y axis
    ranges to increase the chance of getting the visuallu pleasing result.
    Each time it saves the plots it overwrites the old plots. Each of these
    plots add the string "_n" to the end of the name (where n is integer
    number).
    """

    # Default visualization properties used in case of missing values.
    default_clr = 'k'
    default_line_style = '-'
    default_line_width = 1.0

    def __init__(self, fname, period=10, names=None, colors=None,
                 line_styles=None, xlab='epochs', ylab_data='E',
                 ylab_lr='learning rate', logy_lr=True, title='',
                 legend=True, legend_loc='best', grid=True,
                 xlim=None, ylim_data=None, ylim_lr=None,
                 font_size=12):
        """
        Args:
            fname (str): Absolute path to plot.
            period (int): Interval between saves in epochs.
            names (dict): key - variable name, value a string to display.
            colors (dict): {'varname': 'color_code'}
            line_styles (dict): {'varname': 'style_code'}
            xlab (str): x-axis kabel.
            ylab_data (str): y-axis flor data (left)
            ylab_lr (str): y-axis for learning rate (right)
            logy_lr (bool): Whether to display y-axis in log scale.
            title (str): Title.
            legend (bool): Whether to display legend.
            legend_loc (int or str): See `pytplot.legend`
            grid (bool): Show grid?
            xlim (tuple of float): Limits of x axis.
            ylim_data (tuple of float): Limits of data y axis.
            ylim_lr (tuple of float): Limits of lr y axis.
            font_size (int): Font size.
        """

        # Init parent.
        super(PlotsSaver, self).__init__(period)

        if not os.path.exists(os.path.dirname(fname)):
            raise Exception('The directory "{d}" intended for saving '
                            'the plots does not exist.'.
                            format(d=os.path.dirname(fname)))
        if period < 1:
            raise Exception('The period value must be greater than 0.')

        # Save parameters.
        self._fname = fname

        # Save vis. properties.
        self._names = names
        self._colors = colors
        self._line_styles = line_styles
        self._xlab = xlab
        self._ylab_data = ylab_data
        self._ylab_lr = ylab_lr
        self._logy_lr = logy_lr
        self._title = title
        self._legend = legend
        self._legend_loc = legend_loc
        self._grid = grid
        self._xlim = xlim
        self._ylim_dat = ylim_data
        self._ylim_lr = ylim_lr
        self._font_size = font_size

        # Check file name and possibly create output dir.
        fpath_base = os.path.dirname(fname)
        if not os.path.exists(fpath_base):
            print('WARNING: Output plot file path {} does not exist, '
                  'attempting to create it.'.format(fpath_base))
            jbfs.make_dir(fpath_base)

    def on_epoch_end(self, ep, force_run=False, **kwargs):
        data = kwargs['history']

        if ep % self._period == 0 or force_run:
            # keys = data.keys()
            keys = self._names.keys()

            # If names were not provided, use data keys.
            names = self._names if self._names is not None \
                else {k: k for k in keys}

            # Get other available vis. properties given data keys.
            colors = {k: self._colors.get(k, PlotsSaver.default_clr)
                      for k in keys}
            line_styles = {k: self._line_styles.
                get(k, PlotsSaver.default_line_style) for k in keys}
            line_widths = {k: PlotsSaver.default_line_width for k in keys}

            # Set y axis range for data.
            mv = np.max([np.max(data[k]) for k in keys if k != 'lr'])
            ylim_d1 = self._ylim_dat if self._ylim_dat is not None else mv
            ylim_d2 = ylim_d1 / 2
            ylim_d3 = ylim_d1 / 4

            # Get file names.
            fname, fext = jbfs.split_name_ext(self._fname)
            f1 = fname + '_1.' + fext
            f2 = fname + '_2.' + fext
            f3 = fname + '_3.' + fext

            # Plot 3 plots with different y axis range.
            for fn, yl in zip([f1, f2, f3], [ylim_d1, ylim_d2, ylim_d3]):
                jbv2.plot_tr_curves_lr(names, data, colors=colors,
                                       line_styles=line_styles,
                                       line_widths=line_widths,
                                       xlab=self._xlab,
                                       ylab_data=self._ylab_data,
                                       ylab_lr=self._ylab_lr,
                                       logy_lr=self._logy_lr,
                                       title=self._title,
                                       legend=self._legend,
                                       legend_loc=self._legend_loc,
                                       grid=self._grid,
                                       xlim=self._xlim,
                                       ylim_data=yl, ylim_lr=self._ylim_lr,
                                       font_size=self._font_size,
                                       display=False, save=True, file=fn)

    def get_file_name(self):
        return self._fname

    def set_file_name(self, fn):
        self._fname = fn

    def add_file_name_suffix(self, suff):
        fn = self.get_file_name()
        nam, ext = jbfs.split_name_ext(fn)
        self.set_file_name(nam + suff + '.' + ext)


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    """
    def __init__(self, lr, sess, factor=0.1, patience=10,
                 epsilon=1e-4, min_lr=1e-6, verbose=True):
        """
        Args:
            lr (tf.Variable): Variable containing learning rate.
            sess (tf.Session): TF session.
            factor (float): Value by which to multiply lr.
            patience (int): How many epochs to wait before changing lr.
            epsilon (float): By how much the monitored value needs to be
                improving
            min_lr (float): Bottom threshold for lr.
        """

        super(ReduceLROnPlateau, self).__init__(period=1)

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self._lr = lr
        self._sess = sess
        self._factor = factor
        self._min_lr = min_lr
        self._epsilon = epsilon
        self._patience = patience
        self._verbose = verbose
        self._wait = 0
        self._best = 0
        self._mode = 'min'
        self._monitor_op = None
        self.reset()

    def reset(self):
        """Resets wait counter and cooldown counter.
        """

        self._monitor_op = lambda a, b: np.less(a, b - self._epsilon)
        self._best = np.Inf

        self._wait = 0
        self.lr_epsilon = self._min_lr * 1e-4

    def on_epoch_end(self, ep, force_run=False, **kwargs):
        current = kwargs['redlr_val']

        if self._monitor_op(current, self._best):
            self._best = current
            self._wait = 0
        else:
            if self._wait >= self._patience:
                old_lr = self._sess.run(self._lr)
                if old_lr > self._min_lr + self.lr_epsilon:
                    new_lr = old_lr * self._factor
                    new_lr = max(new_lr, self._min_lr)
                    self._sess.run(self._lr.assign(new_lr))
                    if self._verbose:
                        print('\nEpoch {:d}: reducing learning rate '
                              'to {:.2e}.\n'.format(ep, new_lr))
                    self._wait = 0
            else:
                self._wait += 1

    def get_file_name(self):
        raise NotImplementedError

    def set_file_name(self, _):
        raise NotImplementedError

    def add_file_name_suffix(self, _):
        raise NotImplementedError


class VisNormalsSaver(Callback):
    """ Saves the visualization of predicted normals in terms of RGB images.
    """
    def __init__(self, period, sess, path_imgs, path_normals_gt,
                 path_out_imgs, inp, outp, feeddict, frame='ocv',
                 normalize=True, norm_mode='global'):
        """

        Args:
            path_imgs:
            path_out_imgs:
            inp:
            outp:
            feeddict:
        """
        # Init parent.
        super(VisNormalsSaver, self).__init__(period)

        self._period = period
        self._sess = sess
        self._path_imgs = path_imgs
        self._path_normals_gt = path_normals_gt
        self._path_out_imgs = path_out_imgs
        self._inp = inp
        self._outp = outp
        self._feeddict = feeddict
        self._frame = frame
        self._normalize = normalize
        self._norm_mode = norm_mode

        jbfs.make_dir(path_out_imgs)

    def on_epoch_end(self, ep, force_run=False, **kwargs):
        if ep % self._period == 0 or force_run:
            files_imgs = jbfs.ls(self._path_imgs)
            files_gt = jbfs.ls(self._path_normals_gt)

            for fim, fgt in zip(files_imgs, files_gt):
                img = jbim.load(jbfs.jn(self._path_imgs, fim))
                if self._normalize:
                    img = jbim.normalize(img, mode=self._norm_mode)
                ngt = np.load(jbfs.jn(self._path_normals_gt, fgt))['normals']

                fd = {k: v for k, v in self._feeddict.items()}
                fd[self._inp] = img[None]
                npred = self._sess.run(self._outp, feed_dict=fd)[0]

                mask = jbim.get_mask(ngt)
                img_ngt = jbn.normals2img(ngt, frame=self._frame)
                img_npred = jbn.normals2img(npred, frame=self._frame,
                                            mask_fgrd=mask)
                img_cmp = np.concatenate((img_ngt, img_npred), axis=1)
                plt.imsave(jbfs.jn(self._path_out_imgs, fim), img_cmp)

    def get_file_name(self):
        raise NotImplementedError

    def set_file_name(self, _):
        raise NotImplementedError

    def add_file_name_suffix(self, _):
        raise NotImplementedError
