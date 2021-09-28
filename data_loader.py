# Python std.
import abc
from abc import abstractmethod
import logging
import threading
import multiprocessing
import time
try:
    import queue
except ImportError:
    import Queue as queue

# 3rd party
import numpy as np

# Project files.
from . import img as jbim
from . import file_sys as jbfs
from . import helpers as jbh


logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)


def load_n_npz(paths, data_field):
    """ Loads data from .npz files `paths`.

    Args:
        paths (str or list of str): Absolute paths to images.
        data_field (str): Name of the key within .npz file.

    Returns:
        np.array: Loaded data wich shape (N, ) + data_sample.shape,
            N is # of paths in `paths`.
    """

    if isinstance(paths, str):
        paths = [paths]

    sample_aux = np.load(paths[0])[data_field]
    n_files = len(paths)

    sh = sample_aux.shape

    data = np.zeros((n_files, ) + sh, dtype=sample_aux.dtype)

    for i in range(n_files):
        data[i] = np.load(paths[i])[data_field]

    return data


def load_dirs_npz(paths, data_field, with_names=False, verbose=False):
    """ Loads data contained in one or more directories given in `path` which
    contain .npz files and returns them. Function expects that all files are
    of the same shape.

    Args:
        paths (str or list of str): Absolute path(s) to directory(ies).
        data_field (str): Name of the key within .npz file.
        with_names (bool): If true, list of list of file names is returned
            as well.
        verbose (bool): Whether to print loading progress.

    Returns:
        np.array: Data of shape (N, ) + data_sample.shape.
    """
    if isinstance(paths, str):
        paths = [paths]

    files_in_dirs = []
    n_files = 0
    for p in paths:
        files = jbfs.ls(p, 'npz')
        nf = len(files)
        if nf == 0:
            logging.warning('No .npz files found in directory {}'.format(p))
        n_files += nf
        files_in_dirs.append(files)

    if n_files == 0:
        raise Exception('No .npz files found in dirs "{}"'.format(paths))

    # Load one sample and find out its shape.
    sample_aux = None
    for p, fs in zip(paths, files_in_dirs):
        if len(fs) > 0:
            sample_aux = np.load(jbfs.jn(p, fs[0]))[data_field]
            break
    sh = sample_aux.shape

    # Load all data.
    data = np.zeros((n_files, ) + sh, dtype=sample_aux.dtype)
    i = 0
    num_dirs = len(paths)
    if verbose:
        print()
    for pi, (p, files) in enumerate(zip(paths, files_in_dirs)):
        num_files = len(files)
        for fi, f in enumerate(files):
            if verbose:
                print('\rLoading dir {}/{}, file {}/{}'.
                      format(pi + 1, num_dirs, fi + 1, num_files), end='')
            data[i] = np.load(jbfs.jn(p, f))[data_field]
            i += 1

    if verbose:
        print()

    if with_names:
        return data, files_in_dirs
    else:
        return data


def generator_queue(generator, max_que_size=10, wait_time=0.05, num_worker=1,
                    pickle_safe=False):
    """ Builds a queue out of a data generator. If `pickle_safe`, use
    a multiprocessing approach. Else, use threading.

    Args:
        generator:
        max_que_size:
        wait_time:
        num_worker:
        pickle_safe:

    Returns:
    """
    generator_threads = []
    if pickle_safe:
        q = multiprocessing.Queue(maxsize=max_que_size)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()

    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if pickle_safe or q.qsize() < max_que_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(num_worker):
            if pickle_safe:
                # Reset random seed else all children processes
                # share the same seed.
                np.random.seed()
                thread = multiprocessing.Process(target=data_generator_task)
            else:
                thread = threading.Thread(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes.
            for p in generator_threads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop, generator_threads


def load_batch(que, stop):
    """ Helper function to get next data batch

    Args:
        que:
        stop:

    Returns:
    tuple: New batch.
    """
    while not stop.is_set():
        if not que.empty():
            gen_output = que.get()
            break
        else:
            time.sleep(0.01)

    if not hasattr(gen_output, '__len__'):
        stop.set()
        raise ValueError('Unexpected output of generator queue, found: ' +
                         str(gen_output))

    return gen_output


class IteratorDirs(abc.ABC):
    """ Base class for loading data.
    """

    def __init__(self, batch_size=64, shuffle=True, seed=None):
        """
        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle data after each epoch.
            seed (int): Seed for rng.
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._batch_idx = 0
        self._total_batches_seen = 0
        self._lock = threading.Lock()

    @staticmethod
    def _get_data_list(dirs, whitelist):
        """ Lists all the files in dirs, saves them as paths to the list
        and returns this list together with the number of found files.

        Args:
            dirs (lsit of str): Directories.
            whitelist (list of str): Allowed extensions.

        Returns:
        np.array of str: List of file paths.
        int: Number of file paths.
        """
        fnames = []
        n = 0

        for d in dirs:
            files = jbfs.ls(d)
            for f in files:
                for ext in whitelist:
                    if f.lower().endswith('.' + ext):
                        n += 1
                        fnames.append(jbfs.jn(d, f))
        return np.array(fnames), n

    def reset(self):
        self._batch_idx = 0

    def _flow_index(self, n, batch_size, shuffle=True, seed=None,
                    fixed_batch_size=False):
        """ Yields the lists of indices corresponding to data samples.

        Args:
            n (int): # of data samples.
            batch_size (int): Required batch size (may vary depending on
                `fixed_batch_size`)
            shuffle (bool): Whether to randomly shuffle the original array
                of indices of the whole ds.
            seed (int): Seed for rng.
            fixed_bs (bool): If False, then some lists might have smaller length
                than `batch_size`. This happens if len(all_samples) / batch_size
                is not an integer number. If True, the yielded list of indices
                will always have length == batch_size. If
                len(all_samples) / batch_size is not an integer number, then
                last portion of indices is just dropped. This means that if
                `shuffle` == False, some data samples will never be seen!
        """
        # Ensure self._batch_idx is 0.
        self.reset()
        n_total = n
        n_used = ((n // batch_size) * batch_size) if fixed_batch_size else n

        while 1:
            if seed is not None:
                np.random.seed(seed + self._total_batches_seen)

            # Create batch indices for source and target data.
            if self._batch_idx == 0:
                if shuffle:
                    index_array = np.random.permutation(n_total)
                else:
                    index_array = np.arange(n_total)

            current_index = (self._batch_idx * batch_size) % n_used
            if n_used >= current_index + batch_size:
                current_batch_size = batch_size
                self._batch_idx = (self._batch_idx + 1) % \
                                  (int(np.ceil(n_used / batch_size)))
            else:
                current_batch_size = n_used - current_index
                self._batch_idx = 0
            self._total_batches_seen += 1

            yield (index_array[current_index:
                               current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    @abstractmethod
    def next(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_num_samples(self):
        raise NotImplementedError


class IteratorDirsImgsNpz(IteratorDirs):
    """ This class implements the data iterator for the case where we have
    images and corresponding data in .npz files.
    """
    def __init__(self, dirs_imgs, dirs_npz, data_field,
                 batch_size=64, shuffle=True, seed=None,
                 fixed_bs=False, reshape_npz=None):
        """
        Args:
            dirs_imgs (list of st): List of paths to dirs containing images.
            dirs_npz (list of str): List of paths to dirs containing .npz files.
            data_field (str): Name of the key within .npz file.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle data after each epoch.
            seed (int): Seed fo rng.
            fixed_bs (bool): Whether to fix the size of the batch.
        """
        super(IteratorDirsImgsNpz, self).__init__(batch_size, shuffle, seed)
        self._data_field = data_field
        self._reshape_npz = reshape_npz
        self._files_imgs, self._num_imgs = self._get_data_list(dirs_imgs,
                                                               ['tiff', 'tif'])
        self._files_meshes, num_npz = self._get_data_list(dirs_npz,
                                                          ['npz'])

        if self._num_imgs != num_npz:
            raise Exception('Number of images and .npz samples does not '
                            'match, {nx} != {ny}'.
                            format(nx=self._num_imgs, ny=num_npz))

        logging.info('Found {n} images and labels.'.format(n=self._num_imgs))

        self._index_generator = self._flow_index(self._num_imgs,
                                                 self._batch_size,
                                                 shuffle=self._shuffle,
                                                 seed=self._seed,
                                                 fixed_batch_size=fixed_bs)

    def next(self):
        with self._lock:
            index_array, cur_idx, cur_bs = next(self._index_generator)

        batch_imgs = jbim.load_n(self._files_imgs[index_array], norm=True)
        batch_npz = load_n_npz(self._files_meshes[index_array],
                               data_field=self._data_field)
        if self._reshape_npz is not None:
            batch_npz = batch_npz.reshape(self._reshape_npz)

        return batch_imgs, batch_npz

    def get_num_samples(self):
        return self._num_imgs

class IteratorDirsNpz(IteratorDirs):
    """ This class implements the data iterator for the case we have
    just .npz data.
    """
    def __init__(self, dirs_npz, data_field, batch_size=64, shuffle=True,
                 seed=None, fixed_bs=False, reshape=None):
        super(IteratorDirsNpz, self).__init__(batch_size, shuffle, seed)
        self._data_field = data_field
        self._reshape = reshape
        self._files, self._num_files = self._get_data_list(dirs_npz, ['npz'])

        logging.info('Found {} data files.'.format(self._num_files))

        self._index_generator = self._flow_index(self._num_files,
                                                 self._batch_size,
                                                 shuffle=self._shuffle,
                                                 seed=self._seed,
                                                 fixed_batch_size=fixed_bs)

    def next(self):
        with self._lock:
            index_array, cur_idx, cur_bs = next(self._index_generator)
        batch_npz = load_n_npz(self._files[index_array],
                               data_field=self._data_field)
        if self._reshape is not None:
            batch_npz = batch_npz.reshape(self._reshape)
        return batch_npz

    def get_num_samples(self):
        return self._num_files


class DataLoader(abc.ABC):
    """ Base class for the actual data loaders.
    """
    def __init__(self):
        self.dgen_tr = None
        self.dgen_va = None
        self.que_tr = None
        self.stop_tr = None
        self.gen_thread_tr = None
        self.que_va = None
        self.stop_va = None
        self.gen_thread_va = None

    def next_batch_tr(self):
        return load_batch(self.que_tr, self.stop_tr)

    def next_batch_va(self):
        return load_batch(self.que_va, self.stop_va)


class DataLoaderImgsMeshes(DataLoader):
    """ Creates instances of `:class:IteratorDirsImgsNpz` for training,
    validation and test datasets, if specified in `conf` and creates
    generator ques for generating data. Npz files should contain meshes.
    """

    def __init__(self, conf, data_files_mesh='mesh', reshape_mesh=None,
                 verbose=False):
        """
        Args:
            conf (dict): Configuration.
        """
        super(DataLoaderImgsMeshes, self).__init__()

        dspaths = jbh.get_data_paths(conf['ds_name'])
        path_root_imgs = jbfs.jn(dspaths['images'], conf['pathrel_imgs_inp'])
        path_root_meshes = jbfs.jn(dspaths['coords'], conf['pathrel_coords_gt'])

        dirs_imgs_tr = [jbfs.jn(path_root_imgs, d) for d in conf['tr_seqs']]
        dirs_meshes_tr = [jbfs.jn(path_root_meshes, d) for d in conf['tr_seqs']]

        dirs_imgs_va = [jbfs.jn(path_root_imgs, d) for d in conf['va_seqs']]
        dirs_meshes_va = [jbfs.jn(path_root_meshes, d) for d in conf['va_seqs']]

        if verbose:
            logger.info('Images root: {}'.format(path_root_imgs))
            logger.info('Meshes root: {}'.format(path_root_meshes))
            logger.info('Training dirs: {}'.format(conf['tr_seqs']))
            logger.info('Validation dirs: {}'.format(conf['va_seqs']))
            logger.info('Batch size: {}'.format(conf['batch_size']))
            logger.info('Reshape: {}'.format(reshape_mesh))

        self.dgen_tr = IteratorDirsImgsNpz(dirs_imgs_tr, dirs_meshes_tr,
                                           data_files_mesh,
                                           batch_size=conf['batch_size'],
                                           shuffle=True,
                                           reshape_npz=reshape_mesh)
        self.dgen_va = IteratorDirsImgsNpz(dirs_imgs_va, dirs_meshes_va,
                                           data_files_mesh,
                                           batch_size=conf['batch_size'],
                                           shuffle=True,
                                           reshape_npz=reshape_mesh)

        self.que_tr, self.stop_tr, self.gen_thread_tr = \
            generator_queue(self.dgen_tr, max_que_size=128, num_worker=30)
        self.que_va, self.stop_va, self.gen_thread_va = \
            generator_queue(self.dgen_va, max_que_size=128, num_worker=10)


class DataLoaderImgsNormals(DataLoader):
    """ Creates instances of `:class:IteratorDirsImgsNpz` for training,
    validation and test datasets, if specified in `conf` and creates
    generator ques for generating data. Npz files should contain normal maps.
    """

    def __init__(self, conf):
        """
        Args:
            conf (dict): Configuration.
        """
        super(DataLoaderImgsNormals, self).__init__()

        dspaths = jbh.get_data_paths(conf['ds_name'])
        path_root_imgs = jbfs.jn(dspaths['images'], conf['pathrel_imgs_inp'])
        path_root_normals = jbfs.jn(dspaths['normals'],
                                    conf['pathrel_normals_gt'])

        dirs_imgs_tr = [jbfs.jn(path_root_imgs, d) for d in conf['tr_seqs']]
        dirs_normals_tr = [jbfs.jn(path_root_normals, d)
                           for d in conf['tr_seqs']]

        dirs_imgs_va = [jbfs.jn(path_root_imgs, d) for d in conf['va_seqs']]
        dirs_normals_va = [jbfs.jn(path_root_normals, d)
                           for d in conf['va_seqs']]

        self.dgen_tr = IteratorDirsImgsNpz(dirs_imgs_tr, dirs_normals_tr,
                                           'normals',
                                           batch_size=conf['batch_size'],
                                           shuffle=True)
        self.dgen_va = IteratorDirsImgsNpz(dirs_imgs_va, dirs_normals_va,
                                           'normals',
                                           batch_size=conf['batch_size'],
                                           shuffle=True)

        self.que_tr, self.stop_tr, self.gen_thread_tr = \
            generator_queue(self.dgen_tr, max_que_size=128, num_worker=30)
        self.que_va, self.stop_va, self.gen_thread_va = \
            generator_queue(self.dgen_va, max_que_size=128, num_worker=10)


class DataLoaderMeshes(DataLoader):
    """ Creates data loader for meshes in .npz files.
    """
    def __init__(self, conf, shuffle=True, reshape=None, verbose=False):
        """
        Args:
            conf (dict): Configs.
            reshape (tuple): Target shape of loaded data. If None, no reshaping
                is done
            verbose (bool): Whether to print debug messages.
        """
        super(DataLoaderMeshes, self).__init__()

        if verbose:
            logging.info('Loading dataset:\n==============')

        dspaths = jbh.get_data_paths(conf['ds_name'])
        path_root = jbfs.jn(dspaths['coords'], conf['pathrel_meshes'])

        dirs_meshes_tr = [jbfs.jn(path_root, d) for d in conf['tr_seqs']]
        dirs_meshes_va = [jbfs.jn(path_root, d) for d in conf['va_seqs']]

        if verbose:
            logger.info('Dataset root: {}'.format(path_root))
            logger.info('Training dirs: {}'.format(conf['tr_seqs']))
            logger.info('Validation dirs: {}'.format(conf['va_seqs']))
            logger.info('Batch size: {}'.format(conf['batch_size']))
            logger.info('Shuffle: {}'.format(shuffle))
            logger.info('Reshape: {}'.format(reshape))

        self.dgen_tr = IteratorDirsNpz(dirs_meshes_tr, 'mesh',
                                       batch_size=conf['batch_size'],
                                       shuffle=shuffle, reshape=reshape)
        self.dgen_va = IteratorDirsNpz(dirs_meshes_va, 'mesh',
                                       batch_size=conf['batch_size'],
                                       shuffle=shuffle, reshape=reshape)

        if verbose:
            logging.info('Found {} training samples.'.
                         format(self.dgen_tr.get_num_samples()))
            logging.info('Found {} validation samples.'.
                         format(self.dgen_va.get_num_samples()))

        self.que_tr, self.stop_tr, self.gen_thread_tr = \
            generator_queue(self.dgen_tr, max_que_size=128, num_worker=15)
        self.que_va, self.stop_va, self.gen_thread_va = \
            generator_queue(self.dgen_va, max_que_size=128, num_worker=5)


# Tests
if __name__ == "__main__":
    path_ds = '/cvlabdata2/home/jan/projects/jblib/jblib/tests/ds_test_data_loader'

    class IteratorDirsTest(IteratorDirs):
        def __init__(self, dirs_data, data_field='data',
                     batch_size='4', shuffle=True, seed=None,
                     fixed_bs=False):
            super(IteratorDirsTest, self).__init__(batch_size, shuffle, seed)
            self._data_field = data_field
            self._files_data, self._N = self._get_data_list(dirs_data, ['npz'])

            # debug
            print('List of data files ({}):'.format(self._N))
            for f in self._files_data:
                print(f)

            self._index_generator = self._flow_index(self._N, batch_size,
                                                     shuffle=shuffle,
                                                     seed=seed,
                                                     fixed_batch_size=fixed_bs)

        def next(self):
            with self._lock:
                index_array, cur_idx, cur_bs = next(self._index_generator)

            batch_data = load_n_npz(self._files_data[index_array],
                                    data_field=self._data_field)
            return batch_data

        def get_num_samples(self):
            return self._N


    bs = 4
    shuffle = True
    fixed_bs = True
    iters = 1000

    dgen = IteratorDirsTest([path_ds], data_field='data', batch_size=bs,
                            shuffle=shuffle, fixed_bs=fixed_bs)
    que, stop, thred = generator_queue(dgen, max_que_size=10, num_worker=1)

    stats = np.array([0] * 10)
    for i in range(iters):
        batch = load_batch(que, stop)

        print('iter {}'.format(i))
        print('---------')
        for b in batch:
            stats[b] += 1
            print('{}, '.format(b), end='')
        print()

    for i in range(10):
        print('num {}: {}'.format(i, stats[i]))
