# Python std
import os
import yaml
import shutil
import logging

# Project files
from . import file_sys as jbfs


def get_vis_path():
    """ Returns the path to a directory with resources for visualizations.

    Returns:
        str: Absolute path
    """
    envkey = 'SURFREC_VIS'

    if envkey not in os.environ:
        raise Exception('Required environment variable {} is not defined'.
                        format(envkey))

    pth = os.environ[envkey]
    if not os.path.exists(pth):
        raise Exception('Path {} does not exist'.format(pth))

    return pth


def get_training_runs_path():
    """
    Returns:
        str: Absolute path to "training_runs" directory.
    """

    envkey = 'SURFREC_TRAINING_RUNS'

    if envkey not in os.environ:
        raise Exception('Required environment variable {ek} is not defined. '
                        'Please define as the absolute path pointing to the '
                        'directory containing all training runs.'.
                        format(ek=envkey))

    path = os.environ[envkey]

    if not os.path.exists(path):
        logging.warning('The path {p} dose not exist.'.format(p=path))

    return path


def load_conf(path):
    """ Returns the loaded .cfg config file.

    Args:
        path (str): Aboslute path to .cfg file.

    Returns:
    dict: Loaded config file.
    """

    with open(path, 'r') as f:
        conf = yaml.full_load(f)
    return conf


def load_save_conf(path, with_trrun_path=False):
    """ Loads and returns the configuration file (.yaml) and saves (by copying)
    this file into the output subdirectory (within "training_runs") specified
    in the "name_base" argument.

    Args:
        path (str): Absolute path to the configuration file.
        with_trrun_path (bool): If True, returns also tr. run path.

    Returns:
        dict: Loaded config file.
    """

    conf = load_conf(path)
    out_path = jbfs.jn(get_training_runs_path(), conf['pathrel_train_run'],
                       conf['name_base'])

    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = jbfs.unique_dir_name(out_path)
        print('WARNING: The output path {} already exists, creating new dir {}'.
              format(out_path_old, out_path))
        conf['name_base'] = os.path.basename(out_path)

    jbfs.make_dir(out_path)

    # Save.
    shutil.copy(path, jbfs.jn(out_path, os.path.basename(path)))

    if with_trrun_path:
        return conf, out_path
    else:
        return conf


def load_save_conf2(path, fn=None, fn_args={}, key_run='path_train_run',
                    key_name_base='name_base'):
    """ Loads and returns the configuration file (.yaml) and saves (by copying)
    this file into the output directory, which is created using an item
     'key_run' within config file and `fn` function.

    Args:
        path (str): Absolute path to the configuration file.
        fn (callable): Function which takes a config dict as input
            and produces a name for the train run subdirectory.
        fn_args (dict): Named arguments to pass to `fn`.
        key_run (str): Dict key to value storing path to the dir holding
            trianing data.
        key_name_base (str): Dict key to value storing name base.

    Returns:
        conf (dict): Loaded config file.
        out_path (str):
    """

    # Load conf.
    conf = load_conf(path)

    # Get train run subdir path.
    trrun_subdir = conf[key_name_base] if fn is None \
        else fn(conf, **fn_args)
    out_path = jbfs.jn(conf[key_run], trrun_subdir)

    # Create train run dir.
    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = jbfs.unique_dir_name(out_path)
        print('WARNING: The output path {} already exists, creating new dir {}'.
              format(out_path_old, out_path))
        conf[key_name_base] = os.path.basename(out_path)
    jbfs.make_dir(out_path)

    # Save config.
    shutil.copy(path, jbfs.jn(out_path, os.path.basename(path)))

    return conf, out_path


def get_tmp_dir_path():
    """ Gets the path to the tmp directory. This is helpful to save data (like
    trained model) even in case the provided path does not exist (e.g. typo).
    It is expected that environmental variable 'SURFREC_TMPDIR' exists. If not,
    it falls back on standard UNIX '/tmp'.

    Returns:
    str: Absolute path to tmp dir.
    """
    envkey = 'SURFREC_TMPDIR'

    if envkey not in os.environ:
        logging.error('Environmental variable {envkey} not found, falling back '
                      'to default OS\'s tmp dir "/tmp". Please set {envkey} '
                      'variable with the path of your favourite tmp directory '
                      'with write permissions.'.format(envkey=envkey))
        path = '/tmp'
    else:
        path = os.environ[envkey]

    return path


def get_all_datasets_path():
    """
    Returns:
        str: Absolute path to the directory containg all the datasets.
    """
    envkey = 'SURFREC_ALL_DATASETS'

    if envkey not in os.environ:
        raise Exception('Required environmental variabnle {ek} is not defined. '
                        'Please define as the absolute path pointing to the '
                        'directory containing all the datasets.')

    path = os.environ[envkey]

    if not os.path.exists(path):
        logging.warning('The path {p} dose not exist.'.format(p=path))

    return path


def get_data_paths(ds_rel_path):
    """ Returns the map of the absolute paths to the directories related
    to the dataset `ds_rel_path`.

    Args:
        ds_rel_path (str): Relative path to the dataset in question.

    Returns:
        dict: {str: str}, paths to various data of the dataset.
    """
    # Base path to the given dataset.
    ds_base_path = os.path.abspath(jbfs.jn(get_all_datasets_path(), ds_rel_path))

    # Check if the path exists.
    if not os.path.exists(ds_base_path):
        logging.warning('The path {p} does not exist.'.format(p=ds_base_path))

    # Generator.
    blend_files = 'generator/blend'
    blend_textures = 'generator/blend/textures'
    config_camera = 'generator/configs/camera'
    config_lighting = 'generator/configs/lighting'
    config_shapes = 'generator/configs/shapes'
    config_texture = 'generator/configs/texture'
    recordings_camera = 'generator/recordings/camera'
    recordings_lighting = 'generator/recordings/lighting'
    recordings_shapes = 'generator/recordings/shapes'
    recordings_texture = 'generator/recordings/texture'
    sessions = 'generator/sessions'

    # Generated data.
    albedos = 'albedos'
    images = 'images'
    shadings = 'shadings'
    models = 'models'
    labels = 'labels'
    coords = 'coords'
    depth_maps = 'depth_maps'
    corners = 'corners'
    dist_tfs = 'dist_tfs'
    masks = 'masks'
    normals = 'normals'
    videos = 'videos'
    insp_vis = 'inspect_visualize'

    paths = {
        'datasetPath': ds_base_path,
        'blend_files': jbfs.jn(ds_base_path, blend_files),
        'blend_textures': jbfs.jn(ds_base_path, blend_textures),
        'configsCamera': jbfs.jn(ds_base_path, config_camera),
        'configsLighting': jbfs.jn(ds_base_path, config_lighting),
        'configsShapes': jbfs.jn(ds_base_path, config_shapes),
        'configsTexture': jbfs.jn(ds_base_path, config_texture),
        'recordings_camera': jbfs.jn(ds_base_path, recordings_camera),
        'recordings_lighting': jbfs.jn(ds_base_path, recordings_lighting),
        'recordings_shapes': jbfs.jn(ds_base_path, recordings_shapes),
        'recordings_texture': jbfs.jn(ds_base_path, recordings_texture),
        'sessions': jbfs.jn(ds_base_path, sessions),
        'albedos': jbfs.jn(ds_base_path, albedos),
        'images': jbfs.jn(ds_base_path, images),
        'shadings': jbfs.jn(ds_base_path, shadings),
        'models': jbfs.jn(ds_base_path, models),
        'labels': jbfs.jn(ds_base_path, labels),
        'coords': jbfs.jn(ds_base_path, coords),
        'depth_maps': jbfs.jn(ds_base_path, depth_maps),
        'corners': jbfs.jn(ds_base_path, corners),
        'dist_tfs': jbfs.jn(ds_base_path, dist_tfs),
        'masks': jbfs.jn(ds_base_path, masks),
        'normals': jbfs.jn(ds_base_path, normals),
        'videos': jbfs.jn(ds_base_path, videos),
        'insp_vis': jbfs.jn(ds_base_path, insp_vis)
    }

    return paths


def get_learning_data_path():
    """
    Returns:
        str: Absolute path to the directory containing learning data
        (e.g. models, weights, histories).
    """
    envkey = 'SURFREC_LEARNING_DATA'

    if envkey not in os.environ:
        raise Exception('Required environmental variable {ek} is not defined. '
                        'Please define as the absolute path pointing to the '
                        'directory containing all learning data.'.
                        format(ek=envkey))

    path = os.environ[envkey]

    if not os.path.exists(path):
        logging.warning('The path {p} dose not exist.'.format(p=path))

    return path


def get_learning_data_paths():
    """

    Returns:
        dict: {str: str}, Absolute paths to learning data.
    """
    # Base path to the learning data.
    learn_data_base_path = os.path.abspath(get_learning_data_path())

    # Check if the path exists.
    if not os.path.exists(learn_data_base_path):
        logging.warning('The path {p} does not exist.'.
                        format(p=learn_data_base_path))

    models = 'models'
    histories = 'histories'
    weights = 'weights'
    predictions = 'predictions'
    results = 'results'
    plots = 'plots'
    vis = 'vis'

    paths = {
        'models': jbfs.jn(learn_data_base_path, models),
        'histories': jbfs.jn(learn_data_base_path, histories),
        'weights': jbfs.jn(learn_data_base_path, weights),
        'predictions': jbfs.jn(learn_data_base_path, predictions),
        'results': jbfs.jn(learn_data_base_path, results),
        'plots': jbfs.jn(learn_data_base_path, plots),
        'vis': jbfs.jn(learn_data_base_path, vis)
    }

    return paths


def read_samples_list(path_file):
    """ Returns the list of samples read from `path_file` .txt file, where
    each line is expected to contain a (partial) path to a sample. File
    might or might not end with a newline.

    Args:
        path_file (str): Path to .txt file.

    Returns:
        list of str
    """

    with open(path_file, 'r') as f:
        samples = [s[:-1] for s in f.readlines() if len(s) > 0 and s[0] != '\n']

    return samples
