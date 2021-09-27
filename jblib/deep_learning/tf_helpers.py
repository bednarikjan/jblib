""" Various helpers useful when working with Tensorflow.
"""

# 3rd party
import tensorflow as tf

# jblib
import jblib.file_sys as jbfs

# Pyhon
import logging
import re


def initialize_uninitialized_glob(sess):
    """ Initializes only uninitialized global variables within `sess`'s graph.

    Args:
        sess (tf.Session): Session.
    """

    with sess.graph.as_default():
        gvars = tf.global_variables()
        initialized = sess.run([tf.is_variable_initialized(v) for v in gvars])
        vars_to_init = [v for (v, st) in zip(gvars, initialized) if not st]
        sess.run(tf.variables_initializer([v for v in vars_to_init]))


def get_path_conf_weights(path_tr_run):
    """ Finds the weights and config file within `path_tr_run` and returns
    the paths.

    Args:
        path_tr_run (str): Path to trianing run.

    Returns:
        path_conf (str): Path to .yaml config file.
        path_weights (str): Path to weights.
    """

    files_dir = jbfs.ls(path_tr_run)
    path_conf = None
    path_weights = None

    for f in files_dir:
        if f.endswith('yaml'):
            path_conf = jbfs.jn(path_tr_run, f)
        elif f.endswith('meta'):
            path_weights = jbfs.jn(path_tr_run, f[:-5])

    if path_conf is None :
        logging.error('Conf file not found.')
    if path_weights is None:
        logging.error('Weights file not found.')

    return path_conf, path_weights


def get_epoch_from_weights_file(fname):
    """ Extracts the number from the file name of weights. It is expected that
    the file name is already without the extension, e.g. "weights.ckpt-40"

    Args:
        fname (str): File name.

    Returns:
        int: epoch number.
    """

    pattern = '\d+$'
    prog = re.compile(pattern)
    res = prog.search(fname)

    if res is None:
        raise Exception('No number found in the end of file name {}'.
                        format(fname))

    return int(res.group())
