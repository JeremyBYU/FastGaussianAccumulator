from copy import deepcopy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.transform import Rotation as R

COLOR_PALETTE = list(map(colors.to_rgb, plt.rcParams['axes.prop_cycle'].by_key()['color']))

def create_open_3d_mesh(triangles, points, triangle_normals=None, color=COLOR_PALETTE[0]):
    """Create an Open3D Mesh given triangles vertices

    Arguments:
        triangles {ndarray} -- Triangles array
        points {ndarray} -- Points array

    Keyword Arguments:
        color {list} -- RGB COlor (default: {[1, 0, 0]})

    Returns:
        mesh -- Open3D Mesh
    """
    mesh_2d = o3d.geometry.TriangleMesh()
    if points.ndim == 1:
        points = points.reshape((int(points.shape[0] / 3), 3))
    if triangles.ndim == 1:
        triangles = triangles.reshape((int(triangles.shape[0] / 3), 3))
        # Open 3D expects triangles to be counter clockwise
        triangles = np.ascontiguousarray(np.flip(triangles, 1))
    mesh_2d.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_2d.vertices = o3d.utility.Vector3dVector(points)
    if triangle_normals is None:
        mesh_2d.compute_vertex_normals()
        mesh_2d.compute_triangle_normals()
    elif triangle_normals.ndim == 1:
        triangle_normals_ = triangle_normals.reshape((int(triangle_normals.shape[0] / 3), 3))
        # triangles = np.ascontiguousarray(np.flip(triangles, 1))
        mesh_2d.triangle_normals = o3d.utility.Vector3dVector(triangle_normals_)
    mesh_2d.paint_uniform_color(color)
    return mesh_2d

def split_triangles(mesh):
    """
    Split the mesh in independent triangles    
    """
    triangles = np.asarray(mesh.triangles).copy()
    vertices = np.asarray(mesh.vertices).copy()

    triangles_3 = np.zeros_like(triangles)
    vertices_3 = np.zeros((len(triangles) * 3, 3), dtype=vertices.dtype)

    for index_triangle, t in enumerate(triangles):
        index_vertex = index_triangle * 3
        vertices_3[index_vertex] = vertices[t[0]]
        vertices_3[index_vertex + 1] = vertices[t[1]]
        vertices_3[index_vertex + 2] = vertices[t[2]]

        triangles_3[index_triangle] = np.arange(index_vertex, index_vertex + 3)

    mesh_return = deepcopy(mesh)
    mesh_return.triangles = o3d.utility.Vector3iVector(triangles_3)
    mesh_return.vertices = o3d.utility.Vector3dVector(vertices_3)
    mesh_return.paint_uniform_color([0.5, 0.5, 0.5])
    return mesh_return


def assign_vertex_colors(mesh, normal_colors, mask=None):
    """Assigns vertex colors by given normal colors
    NOTE: New mesh is returned

    Arguments:
        mesh {o3d:TriangleMesh} -- Mesh
        normal_colors {ndarray} -- Normals Colors

    Returns:
        o3d:TriangleMesh -- New Mesh with painted colors
    """
    split_mesh = split_triangles(mesh)
    vertex_colors = np.asarray(split_mesh.vertex_colors)
    triangles = np.asarray(split_mesh.triangles)
    if mask is not None:
        triangles = triangles[mask, :]
    for i in range(triangles.shape[0]):
        # import ipdb; ipdb.set_trace()
        color = normal_colors[i, :]
        p_idx = triangles[i, :]
        vertex_colors[p_idx] = color

    return split_mesh


def translate_meshes(mesh_list, current_translation=0.0, axis=0):
    translate_amt = None
    translate_meshes = []
    for mesh_ in mesh_list:
        mesh_ = deepcopy(mesh_)

        bbox = mesh_.get_axis_aligned_bounding_box()
        x_extent = bbox.get_extent()[axis]
        translate_amt = [0, 0, 0]
        translate_amt[axis] = current_translation + x_extent / 2.0
        translate_meshes.append(mesh_.translate(translate_amt, relative=False))
        current_translation += x_extent + 0.5
    return translate_meshes

def plot_meshes(*meshes, shift=True):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.translate([-2.0, 0, 0])
    translate_meshes = []
    current_x = 0.0
    if shift:
        for i, mesh in enumerate(meshes):
            inner_meshes = [mesh]
            if isinstance(mesh, list):
                inner_meshes = mesh
            translate_amt = None
            for mesh_ in inner_meshes:
                mesh_ = deepcopy(mesh_)
                if translate_amt is not None:
                    translate_meshes.append(mesh_.translate(translate_amt, relative=True))
                else:
                    bbox = mesh_.get_axis_aligned_bounding_box()
                    x_extent = bbox.get_extent()[0]
                    translate_amt = [current_x + x_extent / 2.0, 0, 0]
                    translate_meshes.append(mesh_.translate(translate_amt, relative=True))
                    current_x += x_extent + 0.5
    else:
        translate_meshes = meshes

    o3d.visualization.draw_geometries([axis, *translate_meshes])

def draw_normals(normals, line_length=0.05):
    normal_tips = normals * (1 + line_length)
    num_lines = normals.shape[0]
    all_points = np.vstack((normals, normal_tips))
    lines = [[i, i + num_lines] for i in range(num_lines)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_set

def calc_angle_delta(mesh, level):
    normals = np.asarray(mesh.triangle_normals)
    v1 = normals[0, :]
    if level == 0:
        v2 = normals[1, :]
    else:
        v2 = normals[3, :]
    diff = v1 @ v2
    deg = np.rad2deg(np.arccos(diff))
    return deg

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def normalize_box(a:np.ndarray):
    """Normalizes a 2D numpy array of points into unit box [0,1]"""
    min_x = np.min(a[:, 0])
    max_x = np.max(a[:, 0])
    min_y = np.min(a[:, 1])
    max_y = np.max(a[:, 1])
    range_x = max_x - min_x
    range_y = max_y - min_y
    a[:, 0] = (a[:, 0] - min_x) / range_x
    a[:, 1] = (a[:, 1] - min_y) / range_y

    # print(range_x, range_y, min_x)
    # print(a)
    return a


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.allclose(np.abs(a),np.abs(b),atol=.01):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def create_arrow(scale=1):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0,0,0],end=None,vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    # print(end)
    scale = 10; beta = 0; gamma = 0
    T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    T[:3,-1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        mesh = create_arrow(scale)
        axis, angle = align_vector_to_another(b=vec/ scale)
        if axis is None:
            axis_a = axis
        else:
            axis_a = axis * angle
            rotation_3x3 = mesh.get_rotation_matrix_from_axis_angle(axis_a)
    # mesh.transform(T)
    if axis is not None:
        mesh = mesh.rotate(rotation_3x3, center=False)
    mesh.translate(origin)
    return(mesh)

def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def get_pc_all_peaks(peaks, clusters, normals):
    pcd = o3d.geometry.PointCloud()
    peak_normals = np.ascontiguousarray(normals[peaks, :]) * 1.01
    pcd.points = o3d.utility.Vector3dVector(peak_normals)
    colors = get_colors(clusters, colormap=plt.cm.tab20)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def get_point_normals(avg_peaks, avg_weights):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(avg_peaks * 1.01)
    pcd.paint_uniform_color([0, 1, 0])
    meshes = [pcd]

    return meshes

def get_arrow_normals(avg_peaks, avg_weights):
    meshes = []
    for i in range(avg_peaks.shape[0]):
        avg_peak = avg_peaks[i,:]
        mesh = get_arrow(avg_peak, avg_peak * 1.2)
        meshes.append(mesh)

    return meshes


