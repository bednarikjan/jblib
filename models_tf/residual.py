# 3rd party.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Python std.
import logging
import re

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)


class Residual2:
    """ Class implements a ResNet-like architecture, i.e. it is structured
    in stages each consisting of residual blocks, where each consists
    of 2 conv layers and a skip connection (i.e. no bottlenecks are used).
    The first resblock in each stage can have a stride > 1 which reduces the
    spatial size of the feature map. No initial convolutiona and pooling and
    no terminal avg pooling is done.
    """

    # Supported paddings.
    PADDINGS = {'VALID', 'SAME'}

    # Mapping of activation functions to actual tf.nn implementations.
    ACTIVATIONS = {
        'relu': tf.nn.relu
    }

    def __init__(self, input_shape, stages, blocks, filters, kernels, strides,
                 paddings, inp=None, activation='relu', tconv=False,
                 batch_norm=True, is_training=None, last_linear=False,
                 name_scope='residual2', verbose=False):
        """ Builds the model. The input is avaliable as model.inp, output
        as model.outp. `blocks`, `filters`, `kernels`, `strides` and `paddings`
        can be defined as list of the size == `stages`, or a single scalar
        meaning that a given value is the same for all stages.

        Args:
            input_shape (tuple of int): Input shape excluding batch dimension.
            stages (int): Number of stages.
            blocks (int or list of int): Number of blocks within each stage.
                The first block always have conv skip connection, the rest
                have identity skip connection.
            filters (int or list of int): Number of filters used within each
                stage.
            kernels (int or list of int): Sizes of kernels within each stage.
            strides (int or list of int): Strides used in the conv resblocks
                in each stage.
            paddings (str or list of str): Paddings used within each stage,
                one of {'s', 'v'} which correspond to 'SAME' and 'VALID' in
                tf.slim nomenclature.
            inp (tf.Tensor): Optional input tensor. If None, it is created
                as tf.placeholder.
            activation (str): Activation function.
            name_scope (str): Name space for vairables.
            batch_norm (bool): Whether to use batch norm layers.
            is_training (tf.Placeholder or bool): Indicator whether model is
                in trianing mode (used by batch norm layers). If None,
                tf.placeholder is created.
            last_linear (bool): Whether the very last conv layer is passed
                through nonlinearity (ReLU) or it is left as is.
            verbose (bool): Whether to print debug info.
        """

        self._input_shape = input_shape
        self._stages = stages
        self._tconv = tconv
        self._batch_norm = batch_norm
        self._last_lin = last_linear
        self._name_scope = name_scope
        self._verbose = verbose

        self.is_training = is_training
        if self.is_training is None:
            self.is_training = tf.placeholder(tf.bool, shape=(),
                                              name='is_training')

        self._blocks = self._chck_and_cvt2list(blocks, int, stages, 'blocks')
        self._filters = self._chck_and_cvt2list(filters, int, stages, 'filters')
        self._kernels = self._chck_and_cvt2list(kernels, int, stages, 'kernels')
        self._strides = self._chck_and_cvt2list(strides, int, stages, 'strides')
        self._pads = self._chck_and_cvt2list(paddings, str, stages, 'paddings')

        for p in self._pads:
            if p not in self.PADDINGS:
                raise Exception('Unsupported padding "{}".'.format(p))

        if activation not in self.ACTIVATIONS:
            raise Exception('Unsupported activation "{}".'.format(activation))
        self._activ = self.ACTIVATIONS[activation]

        self.inp = inp
        self.outp = None

        # Build the model.
        with tf.variable_scope(self._name_scope):
            self._build()

    @staticmethod
    def _chck_and_cvt2list(val, dtype, reps, name):
        """ Checks whether `val` is a list or a tuple of length `reps` or
        a scalar in which case it creates a list by copying `val` `reps` times.

        Args:
            val: Value to check.
            dtype (type): Expcted data type.
            reps (int): Expected number of items.
            name (str): Name of argument (for error message).

        Returns:
            val (list): List of length `reps`.
        """
        if isinstance(val, dtype):
            val = [val] * reps
        if not isinstance(val, list) and not isinstance(val, tuple):
            raise Exception('"{}" has to be either a scalar or list of type {}'.
                            format(name, dtype.__name__))
        if len(val) != reps:
            raise Exception('"{}" has to be a list of length {}, found {}'.
                            format(name, reps, len(val)))
        return val

    def _resblock_2(self, x, filters, kernel, stride, padding, activ, stage_idx,
                    skip_identity, tconv=False, outp_linear=False,
                    batch_norm=True):
        """ Residual block.

        Args:
            x (tf.Tensor): Input tensor.
            filters (int): Number of filters.
            kernel (int): Size of kernel (square).
            stride (int): Stride (same vertically/horizontally). Only can
                differ from one if skip_identity = False.
            padding (str): One of {'SAME', 'VALID'}
            activ (func): Activtion function.
            stage_idx (int): Index of the stage.
            skip_identity (bool): If True, identity is used for skip connection,
                otherwise a conv layer with `stride`.
            outp_linear (bool): Whether to apply `activation` on the output.
            tconv (bool): Whether to use transposed convolutions.
            batch_norm (bool): Whether to use batch norm layer.

        Returns:
            tf.Tensor: Feature map, output of this residual block.
        """
        if skip_identity and stride != 1:
            logging.warning('Stride cannot be bigger than 1 for identity'
                            'skip connection. Setting stride to 1.')
            stride = 1

        convop = (slim.conv2d, slim.conv2d_transpose)[tconv]

        bn = (None, slim.batch_norm)[batch_norm]
        out_af = (activ, tf.identity)[outp_linear]

        with slim.arg_scope([slim.batch_norm], scale=True,
                            is_training=self.is_training):
            with slim.arg_scope([convop], num_outputs=filters,
                                kernel_size=kernel, padding=padding,
                                normalizer_fn=bn):
                conv1 = convop(x, stride=stride, activation_fn=activ,
                               scope='main_conv1')
                conv2 = convop(conv1, activation_fn=None,
                               scope='main_conv2')
                skip = x
                if not skip_identity:
                    skip = convop(x, stride=stride, activation_fn=None,
                                  scope='skip_conv')
        return out_af(conv2 + skip, name='output'.format(stage_idx))

    def _build(self):
        """ Builds the whole model.
        """
        if self.inp is None:
            self.inp = tf.placeholder(tf.float32,
                                      shape=(None,) + self._input_shape,
                                      name='input')
        x = self.inp
        ns = self._stages
        for (sidx, nb, nf, kern, stride, pad) in \
            zip(range(1, ns + 1), self._blocks, self._filters,
                self._kernels, self._strides, self._pads):

            with tf.variable_scope('stage_{}'.format(sidx)):
                for bidx in range(1, nb + 1):
                    out_lin = self._last_lin and sidx == ns and bidx == nb
                    with tf.variable_scope('block_{}'.format(bidx)):
                        x = self._resblock_2(x, nf, kern, (stride, 1)[bidx > 1],
                                             pad, self._activ, sidx,
                                             skip_identity=bidx > 1,
                                             tconv=self._tconv,
                                             outp_linear=out_lin,
                                             batch_norm=self._batch_norm)

            if self._verbose:
                print('shape after stage {}: {}'.format(sidx, x.shape))

        self.outp = x

    def get_num_params(self):
        """
        Returns:
            int: Number of trainable params.
        """

        matcher = re.compile('(weights:0$|biases:0$|beta:0$|gamma:0$)')
        gvars = tf.global_variables()
        num_pars = [int(np.prod(gv.shape)) for gv in gvars
                    if matcher.search(gv.name)]

        return np.sum(num_pars)


#### Tests.
if __name__ == "__main__":
    logdir = '/cvlabdata2/home/jan/projects/vae/test'

    inp_shape = (2, 2, 3)
    stages = 3
    blocks = [2, 3, 4]
    filters = [2, 4, 8]
    kernels = 3
    strides = 2
    paddings = 'SAME'
    batch_norm = True
    tconv = True

    model = Residual2(inp_shape, stages, blocks, filters, kernels, strides,
                      paddings, activation='relu', tconv=tconv,
                      batch_norm=batch_norm, last_linear=True,
                      name_scope='my_cool_resnet')

    sess = tf.Session()
    writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    outp = sess.run(model.outp, feed_dict={
        model.inp: np.zeros((10, 2, 2, 3), np.float32),
        model.is_training: True
    })

    print(outp.shape)

    num_params = model.get_num_params(sess)
    print('{:.2e}'.format(num_params))

    writer.close()
    sess.close()
