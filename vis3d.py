# import matplotlib
# matplotlib.use('TkAgg')

# Python std.
import base64
import abc
from abc import abstractmethod
import os
import shutil

# 3rd party
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import ipyvolume as ipv

try:
    import plotly.offline as offline
    import plotly.graph_objs as go
    import plotly.figure_factory as FF
except:
    pass

# project files
import jblib.file_sys as jbfs
import jblib.helpers as jbh
import jblib.unit_test as jbut

# Color defines for plotly
RED = 'rgb(250, 50, 50)'
GREEN = 'rgb(50, 250, 50)'
BLUE = 'rgb(50, 50, 250)'
GRAY = 'rgb(128, 128, 128)'


def get_contrast_colors():
    """ Returns 67 contrast colors.

    Returns:
        dict (str -> tuple): Colors, name -> (R, G, B), values in [0, 1].
    """
    clr_names = \
        ['aqua', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral',
         'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod',
         'darkgreen', 'darkmagenta', 'darkolivegreen', 'darkorange',
         'darkorchid', 'darkred', 'darkslateblue', 'darkturquoise',
         'darkviolet', 'deeppink', 'deepskyblue', 'firebrick', 'forestgreen',
         'gold', 'goldenrod', 'green', 'greenyellow', 'hotpink', 'indianred',
         'indigo', 'lawngreen', 'lightsalmon', 'lightseagreen', 'lime',
         'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue',
         'mediumorchid', 'mediumseagreen', 'mediumslateblue',
         'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
         'midnightblue', 'navy', 'olive', 'orange', 'orangered', 'peru',
         'purple', 'rebeccapurple', 'red', 'royalblue', 'saddlebrown', 'salmon',
         'seagreen', 'sienna', 'slateblue', 'springgreen', 'steelblue', 'teal',
         'tomato', 'yellow', 'yellowgreen']
    return {k: mcolors.to_rgb(mcolors.CSS4_COLORS[k]) for k in clr_names}


def visualize_meshes(meshes, simplices, colors=None, opacities=None, title='',
                     title_size=10, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
                     z_range=(-1.0, 1.0), width=800, height=800,
                     plot_mode='web', cam_mode='plotly', **kwargs):
    """ Visualizes multiple meshes as triangulated surfaces using plotly (the
    web browser window should open automatically after calling this function).

    Args:
        meshes (list of np.array of float32): Meshes. Each item is
            (V x 3)-matrix, V is # of vertices of 3 coordinates.
        simplices (list of np.array of int32): Simplices (triangles) describing
            the topology of mesh. Each item is (T x 3)-matrix, T is # of
            triangles, each consisting of 3 indices pointing to `meshes[i]`
            vertices.
        colors (list, optional): Color or colormap for each mesh. For
            description of format see
            `plotly.tools.FigureFactory.create_trisurf()` documentation for
            param `colormap`. If None, then 'rgb(128, 128, 128)' is used for
            all emshes.
        opacities (list of float32, optional): Opacity of each mesh, must take
            value in [0.0, 1.0]. If None, then each mesh will have
            opacity = 1.0.
        title (str, optional): Plot title.
        title_size (int, optional):
        x_range (list of float, optional): Range of [x|y|z]-axis. List must
            contain exactly two values - min and max.
        y_range (list of float, optional):
        z_range (list of float, optional):
        width (int, optional): Plot width in pixels.
        height (int, optional): Plot height in pixels.
        plot_mode (str, optional): The default orientation of the camera, one
            of {'plotly', 'opencv'}:
            plotly: x forward, z up (turntable)
            opencv: z forward, -y up (orbital)
        cam_mode (str, optional): The default orientation of the camera, one
            of {'plotly', 'opencv'}:
            plotly: x forward, z up (turntable)
            opencv: z forward, -y up (orbital)
        **kwargs: Other keyword arguments accepted by plotly's plot function.

    Returns:
        If `plot_mode` == 'divstr', returns :obj:`str` containing `div` HTML
        element.
        If `plot_mode` == 'fig', returns the figyr
    """
    # Check dimensions of simplices lists.
    n = len(meshes)
    if len(simplices) != n:
        raise Exception('Number of simplices lists must equal to the number '
                        'of meshes. {ns} != {nm}.'.
                        format(ns=len(simplices), nm=n))

    # Check number of opacities and if no opacities set, create list of 1.0.
    if opacities is not None:
        if len(opacities) != n:
            raise Exception('Number of opacities must equal to the number '
                            'of meshes. {no} != {nm}.'.
                            format(no=len(opacities), nm=n))
    else:
        opacities = [1.0] * n

    # Check number of colors and if no colors set, create None list.
    if colors is not None:
        if len(colors) != n:
            raise Exception('Number of colors must equal to the number '
                            'of meshes. {nc} != {nm}.'.
                            format(nc=len(colors), nm=n))
    else:
        colors = ['rgb(128, 128, 128)'] * n

    if cam_mode == 'plotly':
        up = dict(x=0, y=0, z=1)
        eye = dict(x=-1, y=0, z=0)
        dragmode = 'turntable'
    elif cam_mode == 'opencv':
        up = dict(x=0, y=-1, z=0)
        eye = dict(x=0, y=0, z=-1)
        dragmode = 'orbit'
    else:
        print('[WARNING]: Unknown cam_mode "{}", falling back to plotly.'.
              format(cam_mode))
        up = dict(x=0, y=0, z=1)
        eye = dict(x=-1, y=0, z=0)
        dragmode = 'turntable'

    # Create figures.
    figures = []
    for i in range(n):
        mesh = meshes[i]
        simp = simplices[i]

        fig = FF.create_trisurf(x=mesh[:, 0], y=mesh[:, 1],
                                z=mesh[:, 2], simplices=simp,
                                colormap=[colors[i]]*2)
        figures.append(fig)

    # Set opacities.
    for i in range(n):
        for d in figures[i].data:
            d.opacity = opacities[i]

    # Set layout
    layout = go.Layout(
        title=title,
        font=dict(family='Courier New, monospace', size=title_size,
                  color='#000000'),
        autosize=True,
        scene=go.Scene(
            xaxis=go.XAxis(range=x_range, autorange=False),
            yaxis=go.YAxis(range=y_range, autorange=False),
            zaxis=go.ZAxis(range=z_range, autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=up,
                center=dict(x=0, y=0, z=0),
                eye=eye
            ),
            dragmode=dragmode
        ),
        height=height,
        width=width
    )

    fig0 = figures[0]
    fig0.layout = layout

    # Merge figures.
    for fig in figures[1:]:
        for d in fig.data:
            fig0.data.append(d)

    # Display
    if plot_mode == 'web':
        offline.plot(fig0, **kwargs)
    elif plot_mode == 'jupyter':
        offline.iplot(fig0, **kwargs)
    elif plot_mode == 'divstr':
        return offline.plot(fig0, output_type='div', **kwargs)
    elif plot_mode == 'fig':
        return fig0
    else:
        raise Exception('Unknonw plot mode "{}". Must be one of '
                        '("web", "divstr", "jupyter", "fig").'.
                        format(plot_mode))


def visualize_meshes_2(meshes, simplices, colors=None, opacities=None, title='',
                     title_size=10, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
                     z_range=(-1.0, 1.0), width=800, height=800,
                     plot_mode='plotly', cam_mode='plotly', **kwargs):

    num_meshes = len(meshes)

    # Resolve colors.
    if colors is not None:
        if len(colors) != num_meshes:
            raise Exception('Number of colors must equal to the number '
                            'of meshes. {nc} != {nm}.'.
                            format(nc=len(colors), nm=num_meshes))
    else:
        colors = [GRAY] * num_meshes

    # Resolve opacities.
    if opacities is not None:
        if len(opacities) != num_meshes:
            raise Exception('Number of opacities must equal to the number '
                            'of meshes. {no} != {nm}.'.
                            format(no=len(opacities), nm=num_meshes))
    else:
        opacities = [1.0] * num_meshes

    # Get and visualize traces.
    traces = []
    for m, s, c, op in zip(meshes, simplices, colors, opacities):
        tr = go.Mesh3d(x=m[:, 0], y=m[:, 1], z=m[:, 2],
                       i=s[:, 0], j=s[:, 1], k=s[:, 2],
                       color=c, opacity=op, showscale=False)
        traces.append(tr)

    vis_traces(traces, title=title, title_size=title_size, x_range=x_range,
               y_range=y_range, z_range=z_range, cam_mode=cam_mode,
               height=height, width=width, plot_mode=plot_mode)


def vis_traces(traces, title='', title_size=10,
               x_range=(-1.0, 1.0), y_range=None, z_range=None,
               cam_mode='plotly', height=800, width=800,
               plot_mode='web', **kwargs):
    """ Visualizes the `traces` using plotly.

    Args:
        traces (list of BasePlotlyType): Traces to plot (e.g. meshes, pclouds)
        title (str): Title to display.
        title_size (int): Size of title font.
        x_range (2-tuple of int): Range of x-axis. If either of `y_range` or
            `z-range` is None, `x_range` is used for all axes.
        y_range (2-tuple of int): Range of y-axis.
        z_range (2-tuple of int): Range of z-axis.
        cam_mode (str): Orientation of the camera, one of {'plotly', 'opencv'}.
        height (int): Canvas height in px.
        width (int): Canvas width in px.
        plot_mode (str): One of {'fig', 'web', 'jupyter', 'divstr'}.
        **kwargs:

    Returns:
        Depending on `plot_mode`:
            'fig' - plotly.garph_objects.Figure
            'web' - None
            'jupyter' - None
            'divstr' - str, string of div HTML tag to embedd the plot in
                webpage.
    """

    # Get camera mode.
    if cam_mode == 'plotly':
        up = dict(x=0, y=0, z=1)
        eye = dict(x=-1, y=0, z=0)
        dragmode = 'turntable'
    elif cam_mode == 'opencv':
        up = dict(x=0, y=-1, z=0)
        eye = dict(x=0, y=0, z=-1)
        dragmode = 'orbit'
    else:
        print('[WARNING]: Unknown cam_mode "{}", falling back to plotly.'.
              format(cam_mode))
        up = dict(x=0, y=0, z=1)
        eye = dict(x=-1, y=0, z=0)
        dragmode = 'turntable'

    # Create a layout.
    layout = go.Layout(
        title=title,
        font=dict(family='Courier New, monospace', size=title_size,
                  color='#000000'),
        autosize=True,
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(range=x_range, autorange=False),
            yaxis=go.layout.scene.YAxis(range=y_range, autorange=False),
            zaxis=go.layout.scene.ZAxis(range=z_range, autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=up,
                center=dict(x=0, y=0, z=0),
                eye=eye
            ),
            dragmode=dragmode
        ),
        height=height,
        width=width
    )

    fig = go.Figure(data=traces, layout=layout)

    if plot_mode == 'web':
        offline.plot(fig, **kwargs)
    elif plot_mode == 'jupyter':
        offline.iplot(fig)
    elif plot_mode == 'divstr':
        return offline.plot(fig, output_type='div', **kwargs)
    elif plot_mode == 'fig':
        return fig
    else:
        raise Exception('Unknonw plot mode "{}". Must be one of '
                        '("web", "divstr", "jupyter", "fig").'.
                        format(plot_mode))


def visualizeMeshesAndPointClouds(meshes, simplices, pointClouds,
                                  colorsMeshes=None, colorsPclouds=None,
                                  opacitiesMeshes=None, opacitiesPclouds=None,
                                  pointSizes=None, title='',
                                  xRange=(-1.0, 1.0), yRange=(-1.0, 1.0), zRange=(-1.0, 1.0),
                                  width=800, height=800, plotMode='web', **kwargs):
    """ Visualizes multiple meshes and point clouds.

    Parameters
    ----------
    meshes
    simplices
    pointClouds
    colorsMeshes
    colorsPclouds
    opacitiesMeshes
    opacitiesPclouds
    title
    xRange
    yRange
    zRange
    width
    height
    pointSize
    plotMode
    kwargs

    Returns
    -------

    """

    numMesh = len(meshes)
    numPcloud = len(pointClouds)

    # Check dimensions of simplices lists.
    if len(simplices) != numMesh:
        raise Exception('Number of simplices lists must equal to the number of meshes. {ns} != {nm}.'.
                        format(ns=len(simplices), nm=numMesh))

    # Check number of opacities and if no opacities set, create list of 1.0.
    if opacitiesMeshes is not None:
        if len(opacitiesMeshes) != numMesh:
            raise Exception('Number of opacities must equal to the number of meshes. {no} != {nm}.'.
                            format(no=len(opacitiesMeshes), nm=numMesh))
    else:
        opacitiesMeshes = [1.0] * numMesh

    if opacitiesPclouds is not None:
        if len(opacitiesPclouds) != numPcloud:
            raise Exception('Number of opacities must equal to the number of point clouds. {no} != {np}.'.
                            format(no=len(opacitiesPclouds), np=numPcloud))
    else:
        opacitiesPclouds = [1.0] * numPcloud

    # Check number of colors and if no colors set, create None list.
    if colorsMeshes is not None:
        if len(colorsMeshes) != numMesh:
            raise Exception('Number of colors must equal to the number of meshes. {nc} != {nm}.'.
                            format(nc=len(colorsMeshes), nm=numMesh))
    else:
        colorsMeshes = ['rgb(128, 128, 128)'] * numMesh

    if colorsPclouds is not None:
        if len(colorsPclouds) != numPcloud:
            raise Exception('Number of colors must equal to the number of point clouds. {nc} != {np}.'.
                            format(nc=len(colorsPclouds), np=numPcloud))
    else:
        colorsPclouds = ['rgb(128, 128, 128)'] * numPcloud

    # Check number of point sizes.
    if pointSizes is not None:
        if len(pointSizes) != numPcloud:
            raise Exception('Number of point sizes must equal to the number of pointclouds. {nps} != {np}.'.
                            format(nps=len(pointSizes), np=numPcloud))
    else:
        pointSizes = [2.0] * numPcloud

    # Create figures.
    figures = []
    for i in range(numMesh):
        mesh = meshes[i]
        simp = simplices[i]
        fig = FF.create_trisurf(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2], simplices=simp,
                                colormap=(colorsMeshes[i], colorsMeshes[i]))
        figures.append(fig)

    # Set opacities.
    for i in range(numMesh):
        for d in figures[i].data:
            d.opacity = opacitiesMeshes[i]

    # Create point clouds data.
    traces = []
    for pc, color, psize, op in zip(pointClouds, colorsPclouds, pointSizes, opacitiesPclouds):
        pcloud = go.Scatter3d(
            x=pc[:, 0],
            y=pc[:, 1],
            z=pc[:, 2],
            mode='markers',
            marker=dict(
                color=color,
                size=psize,
                symbol='circle',
                line=dict(
                    color='rgb(25, 25, 25)',
                    width=1
                ),
                opacity=op
            )
        )
        traces.append(pcloud)

    # Set layout
    layout = go.Layout(
        title=title,
        font=dict(family='Courier New, monospace', size=10, color='#000000'),
        autosize=True,
        scene=go.Scene(
            zaxis=go.ZAxis(range=zRange, autorange=False),
            xaxis=go.XAxis(range=xRange, autorange=False),
            yaxis=go.YAxis(range=yRange, autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        height=height,
        width=width
    )

    fig0 = figures[0]
    fig0.layout = layout

    # Merge figures for meshes.
    for fig in figures[1:]:
        for d in fig.data:
            fig0.data.append(d)

    # Add pointclouds.
    for t in traces:
        fig0.data.append(t)

    # Display
    if plotMode == 'web':
        plotly.offline.plot(fig0, filename='/tmp/temp-plot.html', **kwargs)
    elif plotMode == 'jupyter':
        iplot(fig0, **kwargs)
    elif plotMode == 'divstr':
        return plotly.offline.plot(fig0, output_type='div', **kwargs)
    else:
        raise Exception('Unknonw plot mode "{}". Must be one of ("web", "divstr").'.
                        format(plotMode))


class HtmlVis(abc.ABC):
    """ Base class for building and saving .html file containing plotly plots.
    """

    def __init__(self, html_file='mesh_seq.html', input_img='input_img.png',
                 embed_img=True, copy_js=True):
        """
        Args:
            html_file (str): Absolute path to output .html file.
            input_img (str): Absolute path to input image.
            embed_img (bool): Whether to embedd the image in .html file using
                base64 coding or just to link the standalone image (where link
                is `inputImg` path).
            copy_js (bool): Whether to copy plotly and jquery .js files
                to the output directory.
        """
        self._html_file = jbfs.unique_file_name(html_file)
        self._copy_js = copy_js

        if copy_js:
            path_visres = jbh.get_vis_path()
            self._path_plotlyjs = jbfs.jn(path_visres, 'plotly_script.js')

        if input_img is not None:
            if embed_img:
                self._imgSrcStr = 'data:image/png;base64,' + \
                                  self._png2base64str(input_img)
            else:
                self._imgSrcStr = input_img
        else:
            self._imgSrcStr = ''
        self._plots_str = ''

    @abstractmethod
    def _build_html_str(self):
        pass

    def _png2base64str(self, img_file):
        """ Returns base64 coding of `imgFile`. Since we want to include
        .png in the .html file, if the image has other extension, it
        is converted as saved as .png in tmp folder first.

        Args:
            img_file (str): Abs. path to image file.

        Returns:
            str: Byte string encoding the image using base64 coding.
        """
        # Only png is supported, therefore it must be converted first.
        remove = False
        if not img_file.endswith('.png'):
            img = Image.open(img_file)
            name, _ = jbfs.split_name_ext(os.path.basename(img_file))
            img_file = os.path.join(jbh.get_tmp_dir_path(), name + '.png')
            img.save(img_file)
            remove = True

        with open(img_file, 'rb') as f:
            base64str = base64.b64encode(f.read())

        if remove:
            os.remove(img_file)

        return base64str.decode('ASCII')

    def add_plot_meshes(self, meshes, simplices, colors=None, opacities=None,
                        title='', x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
                        z_range=(-1.0, 1.0), width=800, height=800,
                        cam_mode='plotly'):
        """ Adds plotly 3D interactive plot as a 'div' string in the .html file.

        Args:
            meshes (list of np.array): Each item is (V x 3)-matrix, V is
                # of 3D vertices. Meshes to be visualized.
            simplices (list of np.array): Each item is (T x 3)-matrix, T is
                # of triangles. Triangulation.
            colors (list of str): Colors to be used for meshes in format
                'rgb(R,G,B)', where R, G, B are int32 in [0, 255].
            opacities (list of float32): Opacities of the meshes. Each value
                in [0.0, 1.0].
            title (str): Plot title.
            x_range (list or tuple): 2-tuple, range of x-axis
            y_range (list or tuple): 2-tuple, range of y-axis
            z_range (list or tuple): 2-tuple, range of z-axis
            width (int): Plot width.
            height (int): Plot height.
        """
        self._plots_str += visualize_meshes(meshes, simplices, colors=colors,
                                            opacities=opacities, title=title,
                                            x_range=x_range, y_range=y_range,
                                            z_range=z_range, width=width,
                                            height=height, plot_mode='divstr',
                                            show_link=False,
                                            include_plotlyjs=False,
                                            cam_mode=cam_mode)

        with open(self._html_file, 'w') as f:
            f.write(self._build_html_str())

        if self._copy_js:
            shutil.copy(self._path_plotlyjs, os.path.dirname(self._html_file))


class HtmlVisMesh(HtmlVis):
    """ creates an .html file containing a plotly 3D plot of mesheses.
    The screen is divided into left pane - plotly interactive 3D plot, and
    right pane - input image.
    """

    def __init__(self, html_file='mesh_seq.html', input_img='input_img.png',
                 embed_img=True, copy_js=True):
        super(HtmlVisMesh, self).__init__(html_file, input_img, embed_img,
                                          copy_js)

        self._htmlPart1 = '<html><head><metacharset="utf-8"/><style>#wrapper' \
                          '{width:1100px;overflow:hidden;}.plotly-graph-div' \
                          '{width:800px;float:left;}#input_image{width:300px;' \
                          'overflow:hidden;padding-top:300px;}</style></head>' \
                          '<body><script type="text/javascript" ' \
                          'src="plotly_script.js"></script><div id="wrapper">'
        self._htmlPart3 = '<div id="input_image"><img src="{img}" style=' \
                          '"width:224px;height:224px;" ></div></div></body>' \
                          '</html>'.format(img=self._imgSrcStr)

    def _build_html_str(self):
        return self._htmlPart1 + self._plots_str + self._htmlPart3


class MeshVisMPL:
    """ Meshes visualizer based on matplotlib.pyplot. This is not an ideal
    visualizer since plt is buggy in terms of z-ordering, but it's possible
    to extract the rendered image as an numpy array (as opposed to plotly).
    """
    def __init__(self, figsize=(5, 5), dpi=100, azi=30.0, ele=30.0,
                 show_axes=True, xlabel='x', ylabel='y', zlabel='z',
                 xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0),
                 auto_axes_lims=False, ax_margin=0.05,
                 color_faces=(0., 1., 0.), alpha_faces=1.0,
                 color_edges=(0., 0., 0.), alpha_edges=1.0,
                 pcloud=False, marker_size=1.):
        """
        Args:
            azi (float): Default camera azimuth.
            ele (float): Default camera elevation.
            show_axes (bool): Whether to show grid and exes.
            xlabel (str): x-axis label.
            ylabel (str): y-axis label.
            zlabel (str): z-axis label.
            xlim (2-tuple of float): x-axis limits.
            ylim (2-tuple of float): y-axis limits.
            zlim (2-tuple of float): z-axis limits.
            auto_axes_lims (bool): Whether to compute axes limits automatically.
            color_faces (3-tuple): RGB color of faces, range [0.0, 1.0].
            alpha_faces (float): Faces alpha, 0.0 - transparent, 1.0 - opaque.
            color_edges (3-tuple): RGB color of edges, range [0.0, 1.0].
            alpha_edges (float): Edges alpha, 0.0 - transparent, 1.0 - opaque.
            pcloud (bool): If True, only point cloud is visualized (no faces).
        """

        self._def_azi = azi
        self._def_ele = ele
        self._show_axes = show_axes
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel
        self._def_xlim = xlim
        self._def_ylim = ylim
        self._def_zlim = zlim
        self._auto_axes_lims = auto_axes_lims
        self._ax_margin = ax_margin

        self._color_faces = color_faces
        self._alpha_faces = alpha_faces
        self._color_edges = color_edges
        self._alpha_edges = alpha_edges

        self._pcloud = pcloud
        self._marker_size = marker_size

        self._img_size = np.array(figsize) * dpi

        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._init()

    def _init(self):
        self._ax = self._fig.add_subplot(1, 1, 1, projection='3d')

        if not self._show_axes:
            self._ax.set_axis_off()

        self.set_azi(self._def_azi)
        self.set_ele(self._def_ele)

        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        self._ax.set_zlabel(self._zlabel)

        self._axlims = np.ones((3, 2)) * np.array([1e6, -1e6])
        if not self._auto_axes_lims:
            self._axlims = np.stack([self._def_xlim, self._def_ylim,
                                     self._def_zlim], axis=0)

        if not self._auto_axes_lims:
            self._ax.set_xlim(self._axlims[0, 0], self._axlims[0, 1])
            self._ax.set_ylim(self._axlims[1, 0], self._axlims[1, 1])
            self._ax.set_zlim(self._axlims[2, 0], self._axlims[2, 1])

        # Hacky way to get rid of undesirable margins.
        plt.subplots_adjust(left=-0.2, right=1.2, top=1.2, bottom=-0.2)

    def _update_axes_limits(self, mesh, margin=0.05):
        """ Updates the automatic axes limits given the data, such that
        all the axes would have the same range, the object would be in
        the visualized center and the range corresponds to the axis
        with largest range + margin such that the whole object would
        be visible.

        Args:
            mesh (np.array): Mesh of shape (V, 3).
            margin (float): Percentage of largest axes span to be added to
                the displayed range.
        """
        if margin < 0.0:
            margin = 0.0

        axmin = np.min(mesh, axis=0)
        axmax = np.max(mesh, axis=0)
        rangs = (axmax - axmin) * (1.0 + margin)
        cents = (axmax + axmin) / 2.0
        axmin = cents - (rangs / 2.0)
        axmax = cents + (rangs / 2.0)

        axmin = np.minimum(self._axlims[:, 0], axmin)
        axmax = np.maximum(self._axlims[:, 1], axmax)
        cents = (axmin + axmax) / 2.0
        rmax = np.max(axmax - axmin)

        self._axlims[:, 0] = cents - (rmax / 2.0)
        self._axlims[:, 1] = cents + (rmax / 2.0)

    def add_meshes(self, meshes, tris, colors_faces=None, alphas_faces=None,
                   colors_edges=None, alphas_edges=None):
        """ Adds a meshes to the plot.

        Args:
            meshes (np.array or list of np.array): Mesh(es) of shape (V, 3).
            tris (np.array or list of np.array): Triang(s) of shape (F, 3).
            colors_faces (tuple or list of tuple): Faces colors,
                3-tuple (R, G, B).
            colors_edges (tuple or list of tuple): Edges colors,
                3-tuple (R, G, B).
        """
        def make_list(data, copies=1):
            if not (isinstance(data, list)):
                data = [data] * copies
            return data

        def make_list_sync_len(data, exp_len, name='var'):
            data = make_list(data)
            if len(data) != exp_len:
                if len(data) == 1:
                    data = exp_len * data
                else:
                    raise Exception('"{}" has unexpected length. Expected {}, '
                                    'got {}.'.format(name, exp_len, len(data)))
            return data

        # Unify input data to the lists of same length.
        meshes = make_list(meshes)
        num_meshes = len(meshes)

        # Update axes limits.
        if self._auto_axes_lims:
            for m in meshes:
                self._update_axes_limits(m, margin=self._ax_margin)
            for ax, lim in zip(['x', 'y', 'z'], self._axlims):
                self.set_axlim(ax, lim)

        # if not self._pcloud:
        tris = make_list_sync_len(tris, num_meshes, name='tris')

        colors_faces = (colors_faces, self._color_faces)[colors_faces is None]
        colors_edges = (colors_edges, self._color_edges)[colors_edges is None]
        alphas_faces = (alphas_faces, self._alpha_faces)[alphas_faces is None]
        alphas_edges = (alphas_edges, self._alpha_edges)[alphas_edges is None]

        colors_faces = make_list_sync_len(colors_faces, num_meshes)
        colors_edges = make_list_sync_len(colors_edges, num_meshes)
        alphas_faces = make_list_sync_len(alphas_faces, num_meshes)
        alphas_edges = make_list_sync_len(alphas_edges, num_meshes)

        # Plot meshes.
        for m, tri, cf, af, ce, ae in zip(meshes, tris, colors_faces,
                                          alphas_faces, colors_edges,
                                          alphas_edges):
            if self._pcloud:
                self._ax.scatter(m[:, 0], m[:, 1], m[:, 2], c=cf,
                                 s=self._marker_size, marker='o')
            else:
                m = self._ax.plot_trisurf(m[:, 0], m[:, 1], tri, m[:, 2], shade=True)
                m.set_facecolor(cf + (af, ))
                m.set_edgecolor(ce + (ae, ))

    def clear(self):
        """ Removes all the meshes form the figure.
        """
        self._fig.clear()
        self._init()

    def set_azi(self, azi):
        self._ax.azim = azi

    def set_ele(self, ele):
        self._ax.elev = ele

    def set_axlim(self, ax, lim):
        limf = {
            'x': self._ax.set_xlim,
            'y': self._ax.set_ylim,
            'z': self._ax.set_zlim
        }
        limf[ax](lim[0], lim[1])

    def get_image_as_np_array(self):
        """ Renders the current view and returns it as np array.

        Returns:
            np.array of uint8: Image.
        """
        self._fig.show()
        self._fig.canvas.draw()
        plt.close(self._fig)
        w, h = self._fig.canvas.get_width_height()
        buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
        return buf

    def save_plot(self, path):
        self._fig.savefig(path)

    def show(self):
        self._fig.show()

    def get_img_multi_view(self, eles, azis, text=None, text_pos=(10, 50),
                           font_scale=1.0):
        """ Creates the image consisting of E * A views organized in
        (E, A)-shaped matrix, where E = len(eles), A = len(azis), optionally
        renders a `text` at `text_pos` and returns the image it as an np.array.

        Args:
            eles (list of float): List of elevations.
            azis (list of float): List of azimuths.
            text (str): Text to render. If None, no text is rendered,
            text_pos (2-tuple of int): (x, y) coordinates of bottom left
                corner of the text (top left corner of the image is (0, 0)).
            font_scale (float): Font scale.

        Returns:
            np.array of uint8: Resulting image.
        """

        h, w = self._img_size
        im = np.empty((h * len(eles), w * len(azis), 3), dtype=np.uint8)

        for r in range(len(eles)):
            self.set_ele(eles[r])
            for c in range(len(azis)):
                self.set_azi(azis[c])
                im[r * h: (r + 1) * h, c * w: (c + 1) * w, :] = \
                    self.get_image_as_np_array()

        if text is not None:
            im = cv2.putText(im, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale, color=(0, 0, 0), thickness=2,
                             lineType=cv2.LINE_AA)

        return im


def get_colors_cmap(vals, cmap='jet', mn=None, mx=None):
    """ Assigns a color from color map `cmap` taken from matplotlib
    (https://matplotlib.org/examples/color/colormaps_reference.html) to each
     value from `vals`.

    Args:
        vals (np.array): Values, shape (N, )
        cmap (str): Name of the cmap.
        mn (float): Lower bound. If None, mn = min(vals)
        mx (float): Upper bound. If None, mx = max(vals)
        steps (int): Quantization of color scale.

    Returns:
        np.array: Colors, shape (N, 3).
    """
    # Get lower and upper values bounds.
    mn = np.min(vals) if mn is None else mn
    mx = np.max(vals) if mx is None else mx

    if not hasattr(cm, cmap):
        raise Exception('Unknown colormap "{}".'.format(cmap))

    cmf = getattr(cm, cmap)
    return cmf((np.clip(vals, mn, mx) - mn) /
               (mx - mn))[:, :3].astype(np.float32)


def get_colors_jet(vals, mn=None, mx=None, steps=1000):
    """ Assigns a color from 'jet' color map
    (https://matplotlib.org/examples/color/colormaps_reference.html) to each
     value from `vals`.

    Args:
        vals (np.array): Values, shape (N, )
        mn (float): Lower bound. If None, mn = min(vals)
        mx (float): Upper bound. If None, mx = max(vals)
        steps (int): Quantization of color scale.

    Returns:
        np.array: Colors, shape (N, 3).
    """
    # # Get lower and upper values bounds.
    # mn = (mn, np.min(vals))[mn is None]
    # mx = (mx, np.max(vals))[mx is None]
    #
    # colors = cm.jet(np.linspace(0, 1, steps))[:, :3]
    # inds = np.round((np.clip(vals, mn, mx) - mn) /
    #                 (mx - mn) * (steps - 1)).astype(np.int32)
    # return colors[inds]
    return get_colors_cmap(vals, cmap='jet', mn=mn, mx=mx, steps=steps)


def get_colors2d_fnet(uv, mn=None, mx=None):
    """ Returns the colors sampled from the following color map f: R^2 -> R^3,
    as used in [1] (FoldingNet).

              |r(u, v)|   |1 - u              |
    f(u, v) = |g(u, v)| = |0.3 + 0.35u + 0.35v|
              |b(u, v)|   |1 - v              |

    uv space:
     v
      ^
      |
      +---> u

    [1] Y. Yang et. al. FoldingNet: Point Cloud Auto-encoder via Deep Grid
    Deformation. CVPR 2018. https://arxiv.org/abs/1712.07262

    Args:
        uv (np.array): 2D samples in [0, 1], shape (N, 2).
        mn (tuple): Lower bound of uv range, shape (2, ).
        mx (tuple): Upper bound of uv range, shape (2, ).

    Returns:
        np.array: RGB colors in [0, 1], shape (N, 3).
    """

    mn = (np.array(mn), np.min(uv, axis=0))[mn is None]
    mx = (np.array(mx), np.max(uv, axis=0))[mx is None]
    assert(np.all(np.min(uv, axis=0) >= mn) and
           np.all(np.max(uv, axis=0) <= mx))

    uv = (uv - mn) / (mx - mn)
    return np.stack([1. - uv[:, 0],
                     0.3 + np.sum(0.35 * uv, axis=1),
                     1. - uv[:, 1]], axis=1)


def get_colors2d_fnet_image(w=1000, show=True):
    """ Visualizes and returns the RGB image representing the color coding of
    the UV space as used in `get_colors2d_fnet`.

    Args:
        w (int): Image width/height, [px].
        show (bool): Whether to visualize the image using pyplot.

    Returns:
        np.array of float32: RGB image of shape (w, w, 3).
    """
    pxs = np.stack(np.meshgrid(np.linspace(0., 1., w),
                               np.linspace(0., 1., w)),
                   axis=2).reshape((-1, 2))

    img = get_colors2d_fnet(pxs).reshape((w, w, 3))[::-1]

    if show:
        plt.figure()
        plt.imshow(img)
        plt.xlabel('u')
        plt.ylabel('v')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return img


def ipv_plot_normals(verts, norms, height=1., width=0.01, color='green'):
    """ Plots the vertex normals using ipyvolume library (ipv). ipv does not
      support plotting normals as lines, this is a workaround to it. Each normal
      is represented as 3-faced tetrahedron (the base face is omitted).

    Args:
        verts (np.array): Vertices, shape (V, 3).
        norms (np.array): Vertex normals, shape (V, 3).
        height (float): Height of the tetrahedron.
        width (float): diameter of the circle serving as the tetrahedron's base.
        color (str): Color.
    """
    assert (np.allclose(np.linalg.norm(norms, axis=1), 1.))

    V = verts.shape[0]

    # Get an arrow shape.
    A = np.array([0., 1., 0.]) * width
    B = np.array([np.cos(np.pi / 6.), -np.sin(np.pi / 6.), 0.]) * width
    C = np.array([-np.cos(np.pi / 6.), -np.sin(np.pi / 6.), 0.]) * width
    D = np.array([0., 0., 1.]) * height
    averts = np.stack([A, B, C, D], axis=0)
    afaces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)

    # Find rotation axes and engles.
    vecz = np.array([0., 0., 1.], dtype=np.float32)
    sp = norms @ vecz  # (V, )
    angs = np.arccos(sp)  # (V, )
    rotax = np.cross(vecz, norms)  # (V, 3)

    # Find axes aligned to Z-axis.
    zaligned = np.abs(sp) > 0.999
    rotax[zaligned] = np.array([1., 0., 0.], dtype=np.float32)
    rotax /= np.linalg.norm(rotax, axis=1, keepdims=True)

    # Rotate the tetrahedron by each rotation.
    rots = Rotation.from_rotvec(angs[:, None] * rotax)
    averts_tfd = np.zeros((V, 12), dtype=np.float32)
    for i, v in enumerate(averts):
        averts_tfd[:, i * 3:(i + 1) * 3] = rots.apply(v)
    averts_tfd = averts_tfd.reshape((-1, 3))  # (4V, 3)

    # Move to incident points.
    averts_tfd += np.tile(verts, (1, 4)).reshape((-1, 3))

    # Adjust faces.
    afaces = np.tile(afaces, (V, 1))
    offsets = np.tile((np.arange(V) * 4)[:, None], (1, 3)).reshape((-1, 1))
    afaces = afaces + offsets

    # Plot.
    ipv.plot_trisurf(*averts_tfd.T, afaces, color=color)


################################################################################
### Tests
if __name__ == "__main__":
    ### Test MeshVisMPL:

    import os
    cwd = os.getcwd()
    path_plots = os.path.join(cwd, 'tests/vis3d_test')

    # Make some single-triangle meshes.
    m1 = np.array([
        [0., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]
    ], dtype=np.float32)
    tri1 = np.array([
        [0, 1, 2]
    ], dtype=np.int32)

    m2 = np.array([
        [0., 0., 0.],
        [-1.5, -1.5, 0.],
        [-1.5, -1.5, -1.5]
    ], dtype=np.float32)
    tri2 = tri1

    m3 = np.array([
        [3., 3., 3.],
        [3., 4., 4.],
        [4., 4., 4.]
    ], dtype=np.float32)
    tri3 = tri1

    ############################################################################
    # Test visualizaing meshes.
    jbut.next_test('Visualizing meshes')
    vism1 = MeshVisMPL()
    vism1.add_meshes(m1, tri1)
    vism1.show()

    vism1.add_meshes(m2, tri2, colors_faces=(1, 0, 0))
    vism1.show()

    vism2 = MeshVisMPL(auto_axes_lims=True)
    vism2.add_meshes(m1, tri1)
    vism2.show()
    vism2.add_meshes(m2, tri2, colors_faces=(1, 0, 0))
    vism2.show()
    vism2.add_meshes(m3, tri3, colors_faces=(0, 0, 1))
    vism2.show()

    azis = np.linspace(0, 360, 10)
    eles = [0, 30]

    for e in eles:
        vism2.set_ele(e)
        for a in azis:
            vism2.set_azi(a)
            vism2.show()
            vism2.save_plot(os.path.join(path_plots, 'fr_e{}_a{}.png'.
                                         format(int(e), int(a))))

    import matplotlib.pyplot as plt
    im = vism2.get_image_as_np_array()
    plt.imshow(im)
    plt.show()

    ############################################################################
    ### Test getting multi view img.
    jbut.next_test('Multiview image')
    vism3 = MeshVisMPL(figsize=(5, 5), dpi=50, show_axes=False,
                       auto_axes_lims=True)

    eles = [30, 0, -30]
    azis = [-30, 0, 30]

    vism3.add_meshes(m3, tri3)
    im = vism3.get_img_multi_view(eles, azis, text='sample text')
    plt.imsave(os.path.join(path_plots, 'im_multi.png'), im)

    ############################################################################
    ### Test visualizing pcloud.
    jbut.next_test('Visualizing pointcloud')
    pcloud1 = np.random.uniform(-5.0, 5.0, (200, 3))
    pcloud2 = np.random.uniform(-5.0, 5.0, (200, 3))

    vis_pc = MeshVisMPL(figsize=(5, 5), dpi=50, show_axes=False,
                        auto_axes_lims=True, ax_margin=0., pcloud=True)
    vis_pc.add_meshes([pcloud1, pcloud2], None,
                      colors_faces=[(1, 0, 0), (0, 1, 0)])
    vis_pc.show()

    ############################################################################
    ### Test visualizing pcloud multi image.
    jbut.next_test('Visualizing pointcloud multi image')
    eles = [-30., 30.]
    azis = [-30., 30.]
    img = vis_pc.get_img_multi_view(eles, azis, text='test pclouds')
    plt.imshow(img)
    plt.show()

    ############################################################################
    ### Test find marker size.
    jbut.next_test('Find marker size')

    pcloud1 = np.random.uniform(-0.5, 0.5, (200, 3))
    pcloud2 = np.random.uniform(-0.5, 0.5, (200, 3))

    marker_size = 0.1
    vis_pc2 = MeshVisMPL(figsize=(2, 2), dpi=200, show_axes=False,
                         auto_axes_lims=True, ax_margin=0., pcloud=True,
                         marker_size=marker_size)
    vis_pc2.add_meshes([pcloud1, pcloud2], None,
                      colors_faces=[(1, 0, 0), (0, 1, 0)])
    vis_pc2.add_meshes([pcloud1, pcloud2], None,
                      colors_faces=[(1, 0, 0), (0, 1, 0)])
    img = vis_pc2.get_img_multi_view(eles, azis, text='test pclouds')
    plt.imshow(img)
    plt.show()

    ############################################################################
    ### Test find marker size.
    jbut.next_test('get_colors_cmap')

    cmap = 'cool'
    N = 100
    mn = 0.
    mx = 1.

    vals = np.linspace(mn, mx, N)
    clrs = get_colors_cmap(vals, cmap=cmap, mn=mn, mx=mx)  # (N, 3)
    assert clrs.shape == (N, 3)
    img = np.tile(clrs[None], (50, 1, 1))

    plt.imshow(img)
    plt.show()
