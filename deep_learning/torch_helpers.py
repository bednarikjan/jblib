# 3rd party
import torch
import torch.nn as nn
import numpy as np
import cv2

# Python std
import os
from collections import OrderedDict
import shutil

# project files
from .. import file_sys as jbfs
from .. import vis3d as jbv3
from .. import depth as jbd


class TrainStateSaver:
    """ Saves the training state (weights, optimizer and lr scheduler params)
    to file.

    Args:
        path_file (str): Path to file.
        model (torch.nn.Module): Model from which weights are extracted.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim._LRScheduler): LR scheduler.
        verbose (bool): Whether to print debug info.
    """
    def __init__(self, path_file, model=None, optimizer=None, scheduler=None,
                 batch_sampler=None, verbose=False):
        self._path_file = path_file
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._batch_sampler = batch_sampler
        self._verbose = verbose

        if not os.path.exists(os.path.dirname(path_file)):
            raise Exception('Path "{}" does not exist.'.format(path_file))

        for var, name in zip(
                [model, optimizer, scheduler, batch_sampler],
                ['model', 'optimizer', 'scheduler', 'batch_sampler']):
            if var is None:
                print('[WARNING] TrainStateSaver: {} is None and will not be '
                      'saved'.format(name))

    def get_file_path(self):
        return self._path_file

    def get_file_dir(self):
        return os.path.dirname(self._path_file)

    def __call__(self, file_path_override=None, **kwargs):
        state = kwargs
        if self._model:
            state['weights'] = self._model.state_dict()
        if self._optimizer is not None:
            state['optimizer'] = self._optimizer.state_dict()
        if self._scheduler is not None:
            state['scheduler'] = self._scheduler.state_dict()
        if self._batch_sampler is not None:
            state['batch_sampler'] = self._batch_sampler.state_dict()

        # Get the output file path and save.
        path_file = (file_path_override,
                     self._path_file)[file_path_override is None]

        # Save safely.
        file_exists = os.path.exists(path_file)

        # Create new tmp. name if the output file exists - not to overwrite it.
        pth_tmp = (path_file, jbfs.unique_file_name(path_file))[file_exists]

        # Try to save the file.
        try:
            torch.save(state, pth_tmp)
        except Exception as e:
            print('ERROR: The model weights file {} could not be saved and '
                  'saving is skipped. The exception: "{}"'.
                  format(pth_tmp, e))
            if os.path.exists(pth_tmp):
                os.remove(pth_tmp)
            return

        # Delete the old file and rename the new one to match the required name.
        if file_exists:
            os.remove(path_file)
            os.rename(pth_tmp, path_file)

        if self._verbose:
            print('[INFO] Saved training state to {}'.format(path_file))


# TODO
class PcloudsCmpVisualizer:
    """ Upon being called, visualizes two overlayed 3D pointclouds coming
    from two batches of pclouds. Each image consists of multiple rendered
    views of the pointclouds depending on `eles` and `azis`.

    Args:
        n (int): Max number of images.
        member_img_size (int): Size of one view within the whole image, px.
        azis (list): Azimuths for views, deg.
        eles (list): Elevations for views, deg.
        color_pcloud_a (tuple): Color of the first pcloud.
        color_pcloud_b (tuple): Color of the second pcloud.
    """
    def __init__(self, n, member_img_size=200,
                 azis=(-30., 30.), eles=(-30., 30.),
                 color_pcloud_a=(1, 0, 0), color_pcloud_b=(0, 1, 0),
                 marker_size=0.2, text=None, channels_first=True):

        gs = (len(eles), len(azis))
        self._n = n
        self._clra = np.array(color_pcloud_a)
        self._clrb = np.array(color_pcloud_b)
        self._eles = eles
        self._azis = azis
        self._text = text
        self._channels_first = channels_first

        self._vis = jbv3.MeshVisMPL(figsize=gs, dpi=member_img_size,
                                    show_axes=False, auto_axes_lims=True,
                                    ax_margin=0., pcloud=True,
                                    marker_size=marker_size)

    def __call__(self, batch_a, batch_b, clr_a=None, clr_b=None):
        """
        Args:
            batch_a (np.array): Pclouds a, (B, N, 3), B is batch size.
            batch_b (np.array): Pclouds b, (B, M, 3), B is batch size.
            clr_a (np.array): Color or per-point color for pcloud a,
                shape (3, ) or (B, N, 3).
            clr_b (np.array): Color or per-point color for pcloud b,
                shape (3, ) or (B, M, 3).

        Returns:
            list of np.array: Rendered images.
        """
        B, N = batch_a.shape[:2]
        B, M = batch_b.shape[:2]

        assert(batch_a.shape[0] == batch_b.shape[0])
        assert(clr_a is None or clr_a.shape == (3,) or clr_a.shape == (B, N, 3))
        assert(clr_b is None or clr_b.shape == (3,) or clr_b.shape == (B, M, 3))

        # If a single color is used, it has to have a shape (1, 3), not (3, ).
        if clr_a is None:
            clra = np.stack([self._clra[None]] * B, axis=0)
        else:
            clra = np.stack([clr_a[None]] * B, axis=0) if clr_a.ndim == 1 \
                else clr_a

        if clr_b is None:
            clrb = np.stack([self._clrb[None]] * B, axis=0)
        else:
            clrb = np.stack([clr_b[None]] * B, axis=0) if clr_b.ndim == 1 \
                else clr_b

        imgs = []
        for i in range(np.minimum(self._n, batch_a.shape[0])):
            self._vis.add_meshes([batch_a[i], batch_b[i]], None,
                                 colors_faces=[clra[i], clrb[i]])
            img = self._vis.get_img_multi_view(self._eles, self._azis,
                                               text=self._text)
            if self._channels_first:
                img = img.transpose((2, 0, 1))
            imgs.append(img)
            self._vis.clear()
        return imgs


class DmapsCmpVisualizer:
    """ Upon being called, visualizes GT and pred. depth maps next to each
    other.

    Args:
        n (int): Max number of images.
    """
    def __init__(self, n, mask=True, text=None, text_pos=None):

        self._n = n
        self._mask = mask
        self._text = text
        self._text_pos = text_pos

    def __call__(self, batch_a, batch_b):
        """
        Args:
            batch_a (np.array): Dmaps a, (B, H, W), B is batch size,
                (H, W) is dmap shape.
            batch_b (np.array): Dmaps b, (B, H, W), B is batch size,
                (H, W) is dmap shape.

        Returns:
            list of np.array: Rendered images, each of shape (H, W, 3).
        """
        assert(batch_a.shape[0] == batch_b.shape[0])

        imgs = []
        for i in range(np.minimum(self._n, batch_a.shape[0])):
            dm_gt = batch_a[i]
            dm_p = batch_b[i]

            dmin = np.min(dm_gt[dm_gt != 0])
            dmax = np.max(dm_gt[dm_gt != 0])
            rang = dmax - dmin
            dmin -= 0.1 * rang
            dmax += 0.1 * rang

            mask = jbd.get_mask(dm_gt).astype(np.uint8)[..., None]
            img_gt = jbd.dmap2img(dm_gt, mode='custom', range=(dmin, dmax))
            img_p = jbd.dmap2img(dm_p, mode='custom', range=(dmin, dmax))
            img_p = img_p * mask if self._mask else img_p
            img = np.concatenate([img_gt, img_p], axis=1)
            if self._text is not None:
                tp = (self._text_pos,
                      (10, img.shape[1] // 2 -
                       int(img.shape[1] * 0.1)))[self._text_pos is None]
                img = cv2.putText(
                    img, self._text, tp, cv2.FONT_HERSHEY_SIMPLEX, 1.,
                    color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            img = cv2.putText(img, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.,
                              color=(255, 255, 255), thickness=2,
                              lineType=cv2.LINE_AA)
            img = cv2.putText(img, 'pred', (img_gt.shape[1] + 10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1., color=(255, 255, 255),
                              thickness=2, lineType=cv2.LINE_AA)
            imgs.append(img)
        return imgs


def get_path_tr_state(path_tr_run, ext='tar'):
    """ Finds the training checkpoint file in `path_tr_run` and returns
    its path. The file is expected to be the only one with .`ext` extension.

    Args:
        path_tr_run (str): Path to training run.

    Returns:
        str: Path to train state .`ext` file.
    """

    files_ext = jbfs.ls(path_tr_run, exts=ext)
    if len(files_ext) == 0:
        print(f"[ERROR] Checkpoint (.{ext}) file not found in {path_tr_run}")
    elif len(files_ext) > 1:
        print(f"|[WARNING] More than one .{ext} file found in "
              f"{path_tr_run}. Returning {files_ext[0]}")

    return jbfs.jn(path_tr_run, files_ext[0])


def get_path_conf(path_tr_run):
    """ Finds the conf file in `path_tr_run` and returns its path. The file
    is expected to be the only one with .yaml extension.

    Args:
        path_tr_run (str): Path to training run.

    Returns:
        str: Path to .yaml conf file.
    """

    files_yaml = jbfs.ls(path_tr_run, exts='yaml')

    if len(files_yaml) == 0:
        print('[ERROR] Config file (.yaml) file not found in {}'.
              format(path_tr_run))
    elif len(files_yaml) > 1:
        print('[WARNING] More than one .yaml file found in {}. Returning'
              ' "{}"'.format(path_tr_run, files_yaml[0]))

    return jbfs.jn(path_tr_run, files_yaml[0])


def get_path_conf_tr_state(path_tr_run):
    """ Finds the training checkpoint and config file in `path_tr_run` and
    returns the paths.

    Args:
        path_tr_run (str): Path to training run.

    Returns:
        path_conf (str): Path to .yaml config file.
        path_tr_state (str): Path to .tar training state.
    """

    path_conf = get_path_conf(path_tr_run)
    path_tr_state = get_path_tr_state(path_tr_run)

    return path_conf, path_tr_state


def summary(model, input_size, dtypes=torch.float32, batch_size=-1,
            device='cuda'):
    """ Prints Keras-like summary for a torch model. Code adapted form
    https://github.com/sksq96/pytorch-summary

    Args:
        model (nn.Module): Model.
        input_size (tuple or list of tuples): Size(s) of input tensor(s).
        dtypes (torch.dtype or list of torch.dtype): Dtype(s) of input
            tensor(s).
        batch_size (int): Batch size, if -1, 2 is used.
        device (str): Device, one of {'cuda', 'cpu'}.
    """

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += \
                    torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += \
                    torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
    if not isinstance(dtypes, (tuple, list)):
        dtypes = [dtypes] * len(input_size)

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dt).to(device) for (in_size, dt)
         in zip(input_size, dtypes)]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".\
        format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = np.sum([abs(np.prod(inps) * batch_size *
                                   4. / (1024 ** 2.)) for inps in input_size])
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))# x2 for grads
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")


def has_inf_nan(vals):
    """ Checks whether any of `vals` include inf or nan.

    Args:
        vals (torch.Tensor or list of torch.Tensor): Values to check.

    Returns:
        bool: Whether any inf/nan was found.
    """
    if not isinstance(vals, (tuple, list)):
        if not isinstance(vals, torch.Tensor):
            raise Exception('"vals" must have type torch.Tensor or list of'
                            'torch.Tensor, found "{}"'.format(type(vals)))
        vals = [vals]

    inf_nan_found = False
    for v in vals:
        if not torch.all(torch.isfinite(v)):
            inf_nan_found = True
            break
    return inf_nan_found


class RunningLoss:
    def __init__(self):
        self.reset()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            it = self._its.get(k, 0)
            self._data[k] = self._data.get(k, 0) * (it/(it + 1)) + v/(it + 1)
            self._its[k] = it + 1

    def reset(self):
        self._data = {}
        self._its = {}

    def get_losses(self):
        return self._data.copy()


################################################################################
### Tests
if __name__ == '__main__':
    from .. import unit_test as jbut

    ############################################################################
    ### Test PcloudsCmpVisualizer - clrs
    jbut.next_test('PcloudsCmpVisualizer - clrs')
    import matplotlib.pyplot as plt

    B = 4
    num_imgs = 1
    V = 50

    vis = PcloudsCmpVisualizer(num_imgs)

    # Generate data.
    pc1 = np.random.uniform(-1., 0., (B, V, 3)).astype(np.float32)
    pc2 = np.random.uniform(0., 1., (B, V, 3)).astype(np.float32)

    clrs1 = np.array([1., 0., 0.], dtype=np.float32)
    clrs2 = np.random.uniform(0.5, 0.85, (B, V, 3)).astype(np.float32)

    # imgs = vis(pc1, pc2)
    imgs = vis(pc1, pc2, clr_a=clrs1, clr_b=clrs2)

    assert(len(imgs) == num_imgs)

    plt.figure()
    plt.imshow(imgs[0].transpose((1, 2, 0)))
    plt.show()

    ############################################################################
    ### Test DmapsCmpVisualizer
    jbut.next_test('DmapsCmpVisualizer')
    from .. import file_sys as jbfs
    import matplotlib.pyplot as plt

    num_imgs = 2
    text = 'test_plot'
    # text_pos = (380, 350)
    text_pos = (380, 100)
    path_dmaps = '/cvlabdata1/cvlab/datasets_jan/human_garment/ds_female_tshirt/depth_maps/body_0007'
    path_plots = '/cvlabdata2/home/jan/projects/jblib/jblib/tests/torch_helpers_test'

    vis = DmapsCmpVisualizer(num_imgs, text=text, text_pos=text_pos)

    # Get some depth maps
    dm_files = jbfs.ls(path_dmaps, exts='npz')
    dmaps = np.zeros((len(dm_files), 400, 400), dtype=np.float32)
    for i, dmf in enumerate(dm_files):
        dmaps[i] = np.load(jbfs.jn(path_dmaps, dmf))['depth']

    imgs = vis(dmaps[:4], dmaps[-4:])

    assert(len(imgs) == 2)
    assert(imgs[0].shape == (400, 800, 3))

    fig = plt.figure()
    plt.imshow(imgs[0])
    # fig.savefig(jbfs.jn(path_plots, 'plt.png'))
    plt.show()

    ############################################################################
    ### Test DmapsCmpVisualizer
    jbut.next_test('TrainStateSaver - robustness against save failure')

    pth = '/cvlabdata2/home/jan/projects/cont_param/test/nonexistent'
    f = 'w.tar'
    f2 = 'w2.tar'
    pth_f = jbfs.jn(pth, f)
    pth_f2 = jbfs.jn(pth, f2)
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    list(model.parameters())[0][0, 0] = 1.

    jbfs.make_dir(pth)
    saver = TrainStateSaver(pth_f, model=model, verbose=True)
    shutil.rmtree(pth)
    saver()

    assert not os.path.exists(pth_f)

    jbfs.make_dir(pth)
    saver()

    assert os.path.exists(pth_f)
    assert torch.load(pth_f)['weights']['0.weight'] == 1.

    list(model.parameters())[0][0, 0] = 2.
    saver()
    assert torch.load(pth_f)['weights']['0.weight'] == 2.

    list(model.parameters())[0][0, 0] = 3.
    saver(file_path_override=pth_f2)
    assert os.path.exists(pth_f)
    assert os.path.exists(pth_f2)
    assert torch.load(pth_f)['weights']['0.weight'] == 2.
    assert torch.load(pth_f2)['weights']['0.weight'] == 3.

    list(model.parameters())[0][0, 0] = 4.
    saver()
    assert os.path.exists(pth_f)
    assert os.path.exists(pth_f2)
    assert torch.load(pth_f)['weights']['0.weight'] == 4.
    assert torch.load(pth_f2)['weights']['0.weight'] == 3.

    list(model.parameters())[0][0, 0] = 5.
    shutil.rmtree(pth)
    saver()
    assert not os.path.exists(pth_f)

    list(model.parameters())[0][0, 0] = 6.
    jbfs.make_dir(pth)
    saver()
    assert os.path.exists(pth_f)
    assert torch.load(pth_f)['weights']['0.weight'] == 6.
