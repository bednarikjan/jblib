# 3rd party
import numpy as np

# # In order to import bpy, Blender must by compiled with Python module. Only
# # Pcloud2Dmap class is using bpy in this module, therefore we gently
# # hide the fact that bpy might not have been loaded in order not generate
# # error messages in case Pcloud2Dmap is not needed at all in the first place.
# try:
#     import bpy
# except ImportError:
#     pass

# In order to import pyigl, libigl has to be compiled with Python binding
# support. Only certain functions rely on pyigl, therefore we do not raise
# error when it's not loaded.
try:
    import pyigl as igl
except ImportError:
    pass

# project files
from . import file_sys as jbfs
from . import unit_test as jbut

# Python std
import logging
import itertools


def mesh2obj(mesh, tri):
    """ Converts the triangulated mesh represented as set of 3D vertices
    `mesh` to the string representing the OBJ file content.

    Args:
        mesh (np.array): (V, 3)-matrix, V is # 3D vertices. Mesh.
        tri (np.array): (T, 3)-matrix, T is # triangles. Triangulation.

    Returns:
        str: Content of corresponding OBJ file.
    """

    tri = np.array(tri)
    obj_str = ''

    # Extract vertices coordinates.
    for v in mesh:
        obj_str += 'v {:.6f} {:.6f} {:.6f}\n'.format(*v)

    # Extract faces.
    for t in tri:
        obj_str += 'f {:d} {:d} {:d}\n'.format(*(t + 1))

    return obj_str


def mesh2obj_with_normals(mesh, tri, normals):
    """ Converts the triangulated mesh represented as set of 3D vertices
    `mesh` with corresponding `normals` to the string representing the OBJ
    file content.

    Args:
        mesh (np.array): (V, 3)-matrix, V is # 3D vertices. Mesh.
        tri (np.array or None): (T, 3)-matrix, T is # triangles. Triangulation.
            If None, only 'v' and 'vn' records are written.
        normals (np.array): Per-vertex normal, shape (V, 3).

    Returns:
        str: Content of corresponding OBJ file.
    """

    assert mesh.ndim == 2 and mesh.shape[1] == 3
    assert mesh.shape == normals.shape

    obj_str = ''

    # Extract vertices coordinates.
    for v in mesh:
        obj_str += 'v {:.9f} {:.9f} {:.9f}\n'.format(*v)

    for n in normals:
        obj_str += 'vn {:.9f} {:.9f} {:.9f}\n'.format(*n)

    # Extract faces.
    if tri is not None:
        for t in tri:
            obj_str += f'f {t[0]}//{t[0]} {t[1]}//{t[1]} {t[2]}//{t[2]}\n'

    return obj_str


def load_obj(pth):
    """ Loads mesh from .obj file.

    Args:
        pth (str): Absolute path.

    Returns:
        np.array (float): Mesh, (V, 3).
        np.array (int32): Triangulation, (T, 3).
    """
    with open(pth, 'r') as f:
        lines = f.readlines()

    mesh = []
    tri = []
    for l in lines:
        # vals = l.split(' ')
        vals = l.split()
        if len(vals) > 0:
            if vals[0] == 'v':
                mesh.append([float(n) for n in vals[1:]])
            elif vals[0] == 'f':
                tri.append([int(n.split('/')[0]) - 1 for n in vals[1:]])

    mesh = np.array(mesh, dtype=np.float32)
    tri = np.array(tri, dtype=np.int32)

    return mesh, tri


def load_obj_with_normals(pth):
    """ Loads a model from .obj file. Supports loading the vertices, faces
    and vertex normals. List of vertices and normals are ordered so that
    normals[i] is a vertex normal of vertex vertices[i].

    Args:
        pth (str): Path to an .obj file.

    Returns:
        vertices (np.array of float32): Vertices, shape (V, 3).
        normals (np.array of float32): Vertex normals, shape (V, 3).
        faces (np.array of int32): Faces, shape (F, 3).
    """
    vertices = []
    faces = []
    normals = []
    ninds = {}

    for line in open(pth, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertices.append(v)
        elif values[0] == 'vn':
            v = list(map(float, values[1:4]))
            normals.append(v)
        elif values[0] == 'f':
            face = []
            norms = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]) - 1)
                if len(w) >= 3:
                    norms.append(int(w[2]))
                    ninds[int(w[0]) - 1] = int(w[2]) - 1
            faces.append(face)

    # Convert to np.array
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    normals = np.array(normals, dtype=np.float32)
    ninds = np.array(list(ninds.items()))

    # Reorder normals
    normals = normals[ninds[np.argsort(ninds[:, 0]), 1]]

    # Normalize the normals.
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    assert(vertices.shape == normals.shape)

    return vertices, normals, faces


def procrustes(x_templ, x, scaling=False, reflection=False, gentle=True):
    """ This is procrustes distance code obtained from Bugra. The first matrix
    `x_templ` is the GT data and the `x` is the prediction. For example, if you
    have 17 3D point predictions that you would like to align to the 17 3D
    ground truth positions, `x_gt` and `x` should both be 17x3 matrices.

        TODO: add description of function and params.

    Args:
        x_templ (np.array): Vertices to which `x` will be aligned.
            (V, 3)-matrix, V is # vertices.
        x (np.array): Vertices to align. (V, 3)-matrix, V is # vertices.
        scaling (bool):
        reflection (str):
        gentle (bool): Whether to raise Exception when SVD fails
            (`gentle == False`) or rather to print warning and
            continue with unchanged data (`gentle` == True).

    Returns:

    """

    n, m = x_templ.shape
    ny, my = x.shape
    muX = x_templ.mean(0)
    muY = x.mean(0)
    X0 = x_templ - muX
    Y0 = x - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)

    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        if gentle:
            print('WARNING: SVD failed, returning non-changed data.')
            return None, x, None
        else:
            raise

    V = Vt.T
    T = np.dot(V, U.T)

    if not reflection:
        # does the current solution use a reflection?
        has_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if has_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def project(mesh, K):
    """ Projects mesh vertices to an image, image plane is assumed to be
    XY plane, camera is positioned at origin (0, 0, 0) with
    forward = z, up = -y, left = -x.

    Args:
        mesh (np.array): Mesh of shape (V, 3), V is # vertices. Each row
            should be a point (x, y, z).
        K (np.array): Intrinsic camera matrix, shape (3, 3).

    Returns:
        np.array: 2D points of shape (V, 2).
    """
    mesh_proj = (K @ mesh.T).T
    return mesh_proj[:, :2] / mesh_proj[:, 2][:, None]


def load_dirs_mesh_curv(paths, df_mesh='mesh', df_mean_curv='cmean',
                        df_gauss_curv='cgauss', df_cpmax_curv='cpmax',
                        with_names=False, verbose=False):
    """ Loads the meshes and three types of curvatures from .npz files in
    `paths`.

    Args:
        paths (str or list of str): Paths to dirs containing .npz files.
        df_mesh (str): Data field for meshes.
        df_mean_curv (str): Data field for mean curvature.
        df_gauss_curv (str): Data field for gauss curvature.
        df_cpmax_curv (str): Data field for max principal curvature.
        with_names (bool): Whether to return corresponding file paths.
        verbose (bool): Whether to print out loading info.

    Returns:

    """
    if isinstance(paths, str):
        paths = [paths]

    # Get list of abs. files' paths.
    files = []
    for p in paths:
        files.extend([jbfs.jn(p, f) for f in jbfs.ls(p)])

    num_files = len(files)
    if num_files == 0:
        raise Exception('No files found in dirs {}'.format(paths))

    # Create data structures for meshes and curvatures.
    m_sh = np.load(files[0])[df_mesh].shape
    num_verts = np.prod(m_sh) // 3
    meshes = np.zeros((num_files, ) + m_sh, dtype=np.float32)
    mean_curvs = np.zeros((num_files, num_verts), dtype=np.float32)
    gauss_curvs = np.zeros((num_files, num_verts), dtype=np.float32)
    cpmax_curvs = np.zeros((num_files, num_verts), dtype=np.float32)

    for fidx in range(num_files):
        if verbose:
                print('\rLoading file {}/{}'.
                      format(fidx + 1, num_files), end='')
        data = np.load(files[fidx])
        meshes[fidx] = data[df_mesh]
        mean_curvs[fidx] = data[df_mean_curv].flatten()
        gauss_curvs[fidx] = data[df_gauss_curv].flatten()
        cpmax_curvs[fidx] = data[df_cpmax_curv].flatten()

    if with_names:
        return meshes, mean_curvs, gauss_curvs, cpmax_curvs, files
    else:
        return meshes, mean_curvs, gauss_curvs, cpmax_curvs


def load_n_npz(paths, data_field='mesh', reshape=None):
    """ Loads meshes from .npz `paths` and optionaly reshapes them.

    Args:
        paths (str or list of str): Paths.
        data_field (str): Data field within .npz file.
        reshape (tuple): If not None, each sample will be reshaped.

    Returns:
        np.array: Meshes of shape (N, ) + (data_sample.shape or `reshape`, ).
    """

    if isinstance(paths, str):
        paths = [paths]

    sample_aux = np.load(paths[0])[data_field]
    sh = sample_aux.shape if reshape is None else tuple(reshape)

    n_files = len(paths)

    data = np.zeros((n_files, ) + sh, dtype=sample_aux.dtype)

    for i in range(n_files):
        data[i] = np.load(paths[i])[data_field].reshape(sh)

    return data


def load_dirs_npz(paths, data_field='mesh', reshape=None, with_names=False,
                  verbose=False):
    """ Loads meshes contained in one or more directories given in `path` which
    contain .npz files and returns them. Optionally reshapes the meshes to
    `reshape`.

    Args:
        paths (str or list of str): Absolute path(s) to directory(ies).
        data_field (str): Name of the key within .npz file.
        reshape (tuple): If not None, each sample will be reshaped.
        with_names (bool): If true, list of list of file names is returned
            as well.
        verbose (bool): Whether to print loading progress.

    Returns:
        np.array: Meshes of shape (N, ) + (data_sample.shape or `reshape`, ).
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
    sh = sample_aux.shape if reshape is None else tuple(reshape)

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
            data[i] = np.load(jbfs.jn(p, f))[data_field].reshape(sh)
            i += 1

    if verbose:
        print()

    if with_names:
        return data, files_in_dirs
    else:
        return data


def grid_verts_2d(verts_h, verts_w, size_h, size_w):
    """ Generates a regular 2D grid of vertices.

    Args:
        verts_h (int): Number of vertices in height direction.
        verts_w (int): Number of vertices in width direction.
        size_h (float): Length in height direction.
        size_w (foat): Length in width direction.

    Returns:
        np.array: Grid of shape (verts_h * verts_v, 2)
    """
    return np.stack(np.meshgrid(np.linspace(0., size_w, num=verts_w),
                                np.linspace(0., size_h, num=verts_h))[::-1],
                    axis=2).reshape((-1, 2))


def grid_faces(h, w):
    """ Creates the faces for a mesh with rectangular topology of `h` x `w`
    vertices, where each quad is divided into 2 triangles as follows.

    6    7    8
      x--x--x
      |\ |\ |
      | \| \|
    3 x--x--x 5
      |\ |\ |
      | \| \|
      x--x--x
    0    1    2

    Args:
        h (int): # horizontal verts.
        w (int): # vertical verts.

    Returns:
        np.array of int32: Faces, shape (F, 3), where F = 2 x (`h`-1) x (`v`-1).
    """

    faces = np.zeros((2 * (w - 1), 3), dtype=np.int32)
    for c in range(w - 1):
        faces[2 * c] = [c, c + 1, c + w]
        faces[2 * c + 1] = [c + 1, c + 1 + w, c + w]
    faces = np.tile(faces, (h - 1, 1)) + \
            np.tile((np.arange(0, h - 1, dtype=np.int32) * w)[:, None],
                    (1, 2 * (w - 1))).reshape((-1, 1))
    return faces


class Singleton(type):
    """ Metaclass implementing the Singleton design pattern.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).\
                __call__(*args, **kwargs)
        return cls._instances[cls]


class Pcloud2Dmap(metaclass=Singleton):
    """ Converts the point cloud to depth map of given resolution. Uses
    Blender's internal renderer. The default camera parameters correspond
    to a Kinect camera placed at the origin with z-forward, -x left, -y up.

    Args:
        res_x (int): # horizontal pixels.
        res_y (int): # vertical pixels.
        max_depth (float): Values above `max_depth` are clamped to 0.
        camera (dict): Camera parameters, see DEF_CAMERA for default values.
        remove_cube (bool): Whether to remove the Blender's initial 'Cube'
            object.
    """
    # Default camera settings.
    DEF_CAMERA = {
        'f': 1.87087,
        'sensor_width': 1.73,
        'x': 0.,
        'y': 0.,
        'z': 0.,
        'rot_x': np.pi,
        'rot_y': 0.,
        'rot_z': 0.
    }

    def __init__(self, res_x=224, res_y=224, max_depth=1e3, camera=None,
                 remove_cube=True):
        import bpy

        self._resx = res_x
        self._resy = res_y
        self._max_depth = max_depth

        self._obj = bpy.data.objects.new('model', bpy.data.meshes.new('tmp'))

        # Set scene, renderer.
        scene = bpy.context.scene
        scene.objects.link(self._obj)
        scene.render.resolution_x = res_x
        scene.render.resolution_y = res_y
        scene.render.resolution_percentage = 100
        scene.unit_settings.system = 'METRIC'

        # Set camera.
        cam_settings = (camera, Pcloud2Dmap.DEF_CAMERA)[camera is None]
        cam_cam = bpy.data.cameras['Camera']
        cam_obj = bpy.data.objects['Camera']

        cam_cam.lens = cam_settings['f']
        cam_cam.sensor_width = cam_settings['sensor_width']
        cam_obj.location[0] = cam_settings['x']
        cam_obj.location[1] = cam_settings['y']
        cam_obj.location[2] = cam_settings['z']
        cam_obj.rotation_euler[0] = cam_settings['rot_x']
        cam_obj.rotation_euler[1] = cam_settings['rot_y']
        cam_obj.rotation_euler[2] = cam_settings['rot_z']

        # Remove the inital Cube object.
        if remove_cube and 'Cube' in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects['Cube'])

        # Prepare nodes for depth map rendering.
        scene.use_nodes = True
        tree = scene.node_tree
        links = tree.links

        for n in tree.nodes:
            tree.nodes.remove(n)

        rl = tree.nodes.new('CompositorNodeRLayers')
        viewer = tree.nodes.new('CompositorNodeViewer')
        viewer.use_alpha = False
        links.new(rl.outputs[2], viewer.inputs[0])

    def _set_mesh(self, verts, faces):
        """ Sets the new mesh.

        Args:
            verts (np.array of float): (V, 3)
            faces (np.array of int): (F, 3)
        """
        mesh = bpy.data.meshes.new('new_mesh')
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update()

        mesh_old = self._obj.data
        self._obj.data = mesh

        # Delete the old mesh.
        bpy.data.meshes.remove(mesh_old)

    def get_dmap(self, verts, faces):
        """ Returns the depth map for a mesh given by `verts` and `faces`.

        Args:
            verts (np.array of float): (V, 3)
            faces (np.array of int): (F, 3)

        Returns:
            np.array of float32: Shape (`res_y`, `res_x`).
        """
        self._set_mesh(verts, faces)
        bpy.ops.render.render()
        dmap = np.array(bpy.data.images['Viewer Node'].pixels). \
            reshape((self._resy, self._resx, 4))[::-1, :, 0]  # Revert y-axis
        dmap[dmap > self._max_depth] = 0.
        return dmap


def zero_out_bound_verts(F, vals):
    """ Sets the values to 0 for the boundary vertices.

    Args:
        F (igl.eigen.MatrixXi): Faces, shape (F, 3)
        vals (igl.eigen.MatrixXd): Values, shape (V, D), D >= 1, i.e.
            the whole row is set to zeros.

    Returns:
        C (igl.eigen.MatrixXd): Curvatures with zeros at boundary vertices.
    """

    D = vals.cols()
    bound_verts_inds = igl.eigen.MatrixXi()
    igl.boundary_loop(F, bound_verts_inds)
    zeros = igl.eigen.MatrixXd.Zero(1, D)

    for i in bound_verts_inds:
        vals.setRow(i, zeros)

    return vals


def inv_mass_matrix(V, F):
    """ Computes the inverse of mass matrix.

    Args:
        V (igl.eigen.MatrixXd): Vertices, shape (V, 3).
        F (igl.eigen.MatrixXi): Faces, shape (F, 3)

    Returns:
        Minv (igl.eigen.SparseMatrix): Inverse of mass matrix.
    """
    M = igl.eigen.SparseMatrixd()
    Minv = igl.eigen.SparseMatrixd()
    igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_DEFAULT, M)
    igl.invert_diag(M, Minv)
    return Minv


def principal_cuvratures(V, F):
    """ Returns per-vertex directions and values of minimum and maximum
    curvature.

    Args:
        V (igl.eigen.MatrixXd): Vertices, shape (V, 3).
        F (igl.eigen.MatrixXi): Faces, shape (F, 3)

    Returns:
        pd_max (igl.eigen.MatrixXd): Max. curv. directions, (V, 3).
        pd_min (igl.eigen.MatrixXd): Min. curv. directions, (V, 3).
        pv_max (igl.eigen.MatrixXd): Max. curv. values, (V, 1).
        pv_min (igl.eigen.MatrixXd): Min. curv. values, (V, 1).

    """
    pd_max = igl.eigen.MatrixXd()
    pd_min = igl.eigen.MatrixXd()
    pv_max = igl.eigen.MatrixXd()
    pv_min = igl.eigen.MatrixXd()
    igl.principal_curvature(V, F, pd_max, pd_min, pv_max, pv_min)
    return pd_max, pd_min, pv_max, pv_min


def mean_curvature(V, F, method='quadrics', skip_bound_verts=True):
    """ Computes per-vertex mean curvature using one of two methods:

    1) method='quadrics': Fits the quadrics locally to the surface and
        estimates the principal curvatures. This should result in smoother
        and well-behaved mean cuvrature estimation.
    2) method='lapbelt': Uses discretized Laplace-Beltrami operator weighted
        by cotan weights with voronoi cells.

    Args:
        V (igl.eigen.MatrixXd): Vertices, shape (V, 3).
        F (igl.eigen.MatrixXi): Faces, shape (F, 3)
        method (str): One of {'quadrics', 'lapbelt'}
        skip_bound_verts (bool): If True, the boundary vertices are assigned 0.

    Returns:
        igl.eigen.MatrixXd: (V, 1)-vector of per-vertex mean curvatures.
    """

    # Approach using quadrics fitting.
    if method == 'quadrics':
        _, _, pv_max, pv_min = principal_cuvratures(V, F)
        C = 0.5 * (pv_max + pv_min)
    # Approach using discretized Laplace-Beltrami operator.
    elif method == 'lapbelt':
        # Compute inverse of mass matrix.
        Minv = inv_mass_matrix(V, F)

        # Compute Laplacian and mean curvature normal.
        L = igl.eigen.SparseMatrixd()
        igl.cotmatrix(V, F, L)
        LV = L * V
        HN = -0.5 * Minv * LV

        # Get normals and find positive/negative curvature.
        N = igl.eigen.MatrixXd()
        igl.per_vertex_normals(V, F, N)
        orient = ((np.sum(np.array(HN) * np.array(N), axis=1) >= 0.0)
                  - 0.5) * 2

        # Compute mean curvature
        C = igl.eigen.MatrixXd(np.linalg.norm(HN, axis=1) * orient)
    else:
        raise Exception('Unknwon method "{}". Must be one of {{"quadrics" '
                        ',"lapbelt"\}}'.format(method))

    # Zero-out curvature fo boundary vertices
    if skip_bound_verts:
        C = zero_out_bound_verts(F, C)

    return C


def gauss_curvature(V, F, method='quadrics', skip_bound_verts=True):
    """ Computes the per-vertex guassian curvature using one of two methods:

    1) method='quadrics': Fits the quadrics locally to the surface and
        estimates the principal curvatures. This should result in smoother
        and well-behaved gaussian cuvrature estimation.
    2) method='theoegreg': Uses discretized Theorema Egregium to estimate
        the gaussian curvature for each vertex.

    Args:
        V (igl.eigen.MatrixXd): Vertices, shape (V, 3).
        F (igl.eigen.MatrixXi): Faces, shape (F, 3).
        method (str): One of {'quadrics', 'theoegreg'}.
        skip_bound_verts (bool): If True, the boundary vertices are assigned 0.

    Returns:
        igl.eigen.MatrixXd: (V, 1)-vector of per-vertex gaussian curvatures.
    """
    # Approach using quadrics fitting.
    if method == 'quadrics':
        _, _, pv_max, pv_min = principal_cuvratures(V, F)
        C = pv_max.cwiseProduct(pv_min)
    # Discretized Theorema Egregium divided by the local area.
    elif method == 'theoegreg':
        # Compute inverse of mass matrix.
        Minv = inv_mass_matrix(V, F)

        # Compute curvature, divide by area to get integral average.
        G = igl.eigen.MatrixXd()
        igl.gaussian_curvature(V, F, G)
        C = Minv * G
    else:
        raise Exception('Unknown method "{}". Must be one of {{"quadrics" '
                        ',"theoegreg"\}}'.format(method))

    if skip_bound_verts:
        zero_out_bound_verts(F, C)

    return C


def normals(V, F):
    """ Computes per-vertex normals.

    Args:
        V (igl.eigen.MatrixXd): Vertices, shape (V, 3).
        F (igl.eigen.MatrixXi): Faces, shape (F, 3).

    Returns:
        igl.eigen.MatrixXd: (V, 3)-Per-vertex normals.
    """
    N = igl.eigen.MatrixXd()
    igl.per_vertex_normals(V, F, N)

    return N


def get_nring_boundary_verts(faces, nring=1):
    """ Returns the indices of the vertices whose shortest path to the boundary
    is <= `nring`.
    
    Args:
        faces (np.array of int32): Faces, shape (V, 3).
        nring (int): Length of the maximum shortest path to the boundary.

    Returns:
        np.array of int32: Indices of the vertices.
    """
    # Get adjacency lists.
    F = igl.eigen.MatrixXi(faces)
    A = igl.VectorVectorInt()
    igl.adjacency_list(F, A)
    Anp = np.array(A)

    # Find boundary vertices indices.
    bound = igl.VectorVectorInt()
    igl.boundary_loop(F, bound)

    # Find vertices in the distance <= `nring` from the boundary.
    inds = list(set(itertools.chain.from_iterable([list(it) for it in bound])))
    for i in range(nring - 1):
        inds = list(set(inds).union(
            set(itertools.chain.from_iterable([list(it) for it in Anp[inds]]))))

    return np.array(inds)


def smooth_f_at_mesh_boundary(F, vals, nring=1):
    """ Smooths the function f defined on a mesh close to boundary vertices.
    It looks at `nring` neighborhood of boundary vertices and propagates
    the values of f (`vals`) from n-ring towards 0-ring, i.e. towards the
    boundary vertices. Each vertex of ring-i takes the average value of
    it's 1-neighbors from ring-i+1.

    Example: f is per-vertex mean curvature (cmean) computed over mesh. At the
        boundary vertices the cmean would get higher values (possibly due to
        faulty curvature computation). Thus using this function it is possible
        to smooth it out using the cmean from vertices further away from the
        boundary.

    Args:
        F (igl.eigen.MatrixXi): Faces, shape (V, 3).
        vals (list of np.array): Values of function f, each of shape (V, )
        nring (int): Defines from how far the values should be propagated.

    Returns:
        list of np.array: Adjusted values `vals`.
    """
    if not (isinstance(vals, list) or isinstance(vals, tuple)):
        vals = [vals]
    new_vals = [np.copy(v) for v in vals]
    num_vals = len(new_vals)

    # Get adjacency lists.
    A = igl.VectorVectorInt()
    igl.adjacency_list(F, A)
    Anp = np.array(A)

    # Find boundary vertices indices.
    bound_loops = igl.VectorVectorInt()
    igl.boundary_loop(F, bound_loops)

    # Initialize with boundary vertices.
    rings = {0: set(), 1: list(itertools.chain.from_iterable(
        [list(it) for it in bound_loops]))}  # -1, 0, 1, ...
    sets = {0: set()}  # (-2,-1), (-1,0), (0,1), (1, 2), ...
    adjs = {0: None}

    # Compute the quantities for increasing rings.
    for n in range(1, nring + 1):
        adjs[n] = Anp[rings[n]]
        sets[n] = sets[n - 1].union(set(rings[n]))
        rings[n + 1] = list(set(itertools.chain.from_iterable(
            [list(it) for it in adjs[n]])).difference(sets[n]))

    # Update the vals.
    for r in range(nring, 0, -1):
        for vn, vi in zip(adjs[r], rings[r]):
            neighbs = list(set(vn).difference(sets[r]))

            for i in range(num_vals):
                new_vals[i][vi] = \
                    np.mean(new_vals[i][(vi, neighbs)[len(neighbs) > 0]])

    return new_vals


def area(verts, faces):
    """ Computes the area of the surface as a sum of areas of all the triangles.

    Args:
        verts (np.array of float): Vertices, shape (V, 3).
        faces (np.array of int32): Faces, shape (F, 3)

    Returns:
        float: Total area [m^2].
    """
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    v1s, v2s, v3s = verts[faces.flatten()].\
        reshape((-1, 3, 3)).transpose((1, 0, 2))  # each (F, 3)
    v21s = v2s - v1s
    v31s = v3s - v1s
    return np.sum(0.5 * np.linalg.norm(np.cross(v21s, v31s), axis=1))


################################################################################
### Tests
if __name__ == '__main__':
    import sys

    ############################################################################
    jbut.next_test('grid_faces()')
    h = 5
    w = 3
    faces = grid_faces(h, w)
    print(faces)

    ############################################################################
    if 'bpy' in sys.modules:
        jbut.next_test('Singleton for Pcloud2Dmap')
        dm1 = Pcloud2Dmap(res_x=224, res_y=224)
        dm2 = Pcloud2Dmap(res_x=100, res_y=100)

        assert (dm1._resx == 224 and dm1._resy == 224)
        assert (dm2._resx == 224 and dm2._resy == 224)

        del (dm1)
        del (dm2)
        dm3 = Pcloud2Dmap(res_x=300, res_y=300)
        assert (dm3._resx == 224 and dm3._resy == 224)
        del (dm3)

    ############################################################################
    if 'bpy' in sys.modules:
        jbut.next_test('Pcloud2Dmap() - _set_mesh()')
        import bpy

        dm = Pcloud2Dmap()

        path_render = 'tests/mesh_test/renders'
        V = 100
        F = 200
        iters = 5

        for i in range(iters):
            verts = np.random.uniform(-0.25, 0.25, (V, 3)) + \
                    np.array([0., 0., 1.])
            faces = np.zeros((F, 3), dtype=np.int32)
            for fi in range(F):
                faces[fi] = np.random.permutation(V)[:3]

            dm._set_mesh(verts, faces)

            bpy.context.scene.render.filepath = \
                jbfs.jn(path_render, 'img_{}.png'.format(i))
            bpy.ops.render.render(write_still=True)

    ############################################################################
    if 'bpy' in sys.modules:
        jbut.next_test('Pcloud2Dmap - get_dmap()')

        from . import depth as jbd
        import matplotlib.pyplot as plt

        path_dmap_img = 'tests/mesh_test/renders/dmap.png'

        dmap = dm.get_dmap(verts, faces)
        img = jbd.dmap2img(dmap, mode='minmax_scaled')
        plt.imsave(path_dmap_img, img)

    ############################################################################
    if 'pyigl' in sys.modules:
        ########################################################################
        jbut.next_test('zero_out_bound_verts')
        # 31 x 31 vertices rectangular mesh.
        path_mesh = 'tests/mesh_test/sh-b31_tl_tr-cotton_t-white_01562.obj'

        V = igl.eigen.MatrixXd()
        F = igl.eigen.MatrixXi()
        igl.read_triangle_mesh(path_mesh, V, F)

        vals1 = igl.eigen.MatrixXd.Ones(V.rows(), 1)
        vals1 = np.array(zero_out_bound_verts(F, vals1)).reshape((31, 31))

        vals2 = igl.eigen.MatrixXd.Ones(V.rows(), 3)
        vals2 = np.array(zero_out_bound_verts(F, vals2)).reshape((31, 31, 3))

        assert(np.sum(vals1[[0, -1], :]) + np.sum(vals1[:, [0, -1]]) == 0)
        assert(np.allclose(np.sum(vals2, axis=2) / 3., vals1))

        ########################################################################
        jbut.next_test('mean_curvature()')
        H1 = mean_curvature(V, F, method='quadrics')
        H2 = mean_curvature(V, F, method='lapbelt', skip_bound_verts=True)
        H1np = np.array(H1)
        H2np = np.array(H2)
        print('H1: mean/std: {:.3f}/{:.3f}'.format(np.mean(H1np), np.std(H1np)))
        print('H2: mean/std: {:.3f}/{:.3f}'.format(np.mean(H2np), np.std(H2np)))

        ########################################################################
        jbut.next_test('gauss_curvature()')
        G1 = gauss_curvature(V, F, method='quadrics')
        G2 = gauss_curvature(V, F, method='theoegreg', skip_bound_verts=True)
        G1np = np.array(G1)
        G2np = np.array(G2)
        print('G1: mean/std: {:.3f}/{:.3f}'.format(np.mean(G1np), np.std(G1np)))
        print('G2: mean/std: {:.3f}/{:.3f}'.format(np.mean(G2np), np.std(G2np)))

    igl.adjacency_list()
    igl.VectorVectorInt

    ############################################################################
    jbut.next_test('mesh2obj_with_normals')

    pth = '/Users/janbednarik/research/repos/jblib/jblib/tests/mesh_test/mesh_and_normals.obj'

    N = 10
    F = 20

    verts = np.random.uniform(-1., 1., (N, 3)).astype(np.float32)
    norms = np.random.uniform(-1., 1., (N, 3)).astype(np.float32)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    faces = np.random.randint(0, N, (F, 3)).astype(np.int32)
    objstr = mesh2obj_with_normals(verts, faces, norms)
    print(objstr)

    ############################################################################
    jbut.next_test('grid_verts_2d')

    H = 4
    W = 6
    size_H = 3
    size_W = 5

    gv = grid_verts_2d(H, W, size_H, size_W)

    print(gv)
