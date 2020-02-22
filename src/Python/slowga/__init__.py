import time
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from .helper import assign_vertex_colors, create_open_3d_mesh, calc_angle_delta, plot_meshes, draw_normals
from .icosahedron import refine_icosahedron, create_icosahedron_mesh
from .projections import plot_projection

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../../fixtures/"
EXAMPLE_MESH_1 = str((FIXTURES_DIR / "realsense/example_mesh.ply").absolute())
EXAMPLE_MESH_2 = str((FIXTURES_DIR / "realsense/dense_first_floor_map.ply").absolute())
EXAMPLE_MESH_3 = str((FIXTURES_DIR / "realsense/sparse_basement.ply").absolute())

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [np.identity(3), R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


class GaussianAccumulator(object):
    def __init__(self, gaussian_normals, leafsize=16):
        super().__init__()
        self.nbuckets = gaussian_normals.shape[0]
        self.kdtree = cKDTree(gaussian_normals, leafsize=leafsize)
        self.accumulator = np.zeros(self.nbuckets, dtype=np.float64)
        self.colors = np.zeros_like(gaussian_normals)
        self.gaussian_normals = gaussian_normals

    def integrate(self, normals):
        query_size = normals.shape[0]
        t0 = time.perf_counter()
        _, neighbors = self.kdtree.query(normals)
        t1 = time.perf_counter()
        elapsed_time = (t1 - t0) * 1000
        print("KD tree size: {}; Query Size (K): {}; Execution Time(ms): {:.1f}".format(
            self.nbuckets, query_size, elapsed_time))
        for idx in neighbors:
            self.accumulator[idx] = self.accumulator[idx] + 1

    def normalize(self):
        self.accumulator = self.accumulator / np.max(self.accumulator)
        self.colors = get_colors(self.accumulator, plt.cm.viridis)[:, :3]


def generate_family_of_icosahedron(triangles, vertices, family=[1, 2, 3, 4]):
    meshes = []
    for level in family:
        new_vertices, new_triangles = refine_icosahedron(
            triangles, vertices, level=level)
        new_mesh = create_open_3d_mesh(new_triangles, new_vertices)
        meshes.append(new_mesh)
    return meshes


def generate_sphere_examples():
    ico = create_icosahedron_mesh()
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)

    ico.compute_vertex_normals()
    ico.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()

    return ico, sphere


def visualize_refinement(ico, level=2, plot=False):
    vertices, triangles = refine_icosahedron(np.asarray(
        ico.triangles), np.asarray(ico.vertices), level=level)
    new_mesh = create_open_3d_mesh(triangles, vertices)
    # create lineset of normals
    top_normals = np.asarray(new_mesh.triangle_normals)
    # mask = top_normals[:, 2] >= 0.0
    # new_mesh.remove_triangles_by_mask(~mask)
    # top_normals = np.ascontiguousarray(top_normals[top_normals[:, 2] >= 0.0, :])

    line_set = draw_normals(top_normals)
    pcd_normals = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(top_normals))
    pcd_normals.paint_uniform_color([0.5, 0.5, 0.5])
    if plot:
        plot_meshes(new_mesh, line_set, pcd_normals, shift=False)
    return new_mesh, top_normals


def visualize_gaussian_integration(refined_icosahedron_mesh, gaussian_normals, mesh, ds=50, min_samples=10000, plot=False):
    to_integrate_normals = np.asarray(mesh.triangle_normals)
    num_normals =to_integrate_normals.shape[0]
    ds_normals = int(num_normals/ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    to_integrate_normals = to_integrate_normals[np.random.choice(
        num_normals, to_sample), :]
    ga = GaussianAccumulator(gaussian_normals)
    ga.integrate(to_integrate_normals)
    ga.normalize()
    colored_icosahedron = assign_vertex_colors(
        refined_icosahedron_mesh, ga.colors)
    if plot:
        plot_meshes(colored_icosahedron, mesh)

    return ga, colored_icosahedron


def main():

    ico, sphere = generate_sphere_examples()
    ico_copy = o3d.geometry.TriangleMesh(ico)
    family = [1, 2, 3, 4, 5, 6]
    meshes = generate_family_of_icosahedron(np.asarray(
        ico.triangles), np.asarray(ico.vertices), family)
    family.insert(0, 0)
    meshes.insert(0, ico)
    for level, mesh in zip(family, meshes):
        angle_diff = calc_angle_delta(mesh, level)
        print("Refinement Level: {}; Number of Triangles: {}, Angle Difference: {:.1f}".format(
            level, np.array(mesh.triangles).shape[0], angle_diff))
    meshes.insert(0, sphere)
    plot_meshes(*meshes)
    # Show our chosen refined example
    refined_icosphere, gaussian_normals = visualize_refinement(
        ico_copy, level=3, plot=True)
    # Get an Example Mesh
    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 1:
            continue
        example_mesh = o3d.io.read_triangle_mesh(mesh_fpath)
        example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()
        # plot_meshes(example_mesh)

        ga, colored_icosahedron = visualize_gaussian_integration(
            refined_icosphere, gaussian_normals, example_mesh)
        plot_projection(ga)
        plot_meshes(colored_icosahedron, example_mesh)


if __name__ == "__main__":
    main()


# 'filter_smooth_laplacian', 'filter_smooth_simple',
# for smooth_alg in ['filter_smooth_taubin']:
#     new_mesh = o3d.geometry.TriangleMesh(example_mesh)
#     func = getattr(new_mesh, smooth_alg)
#     new_mesh = func()
#     plot_meshes(new_mesh)
# Visualize Guassiant Integration
