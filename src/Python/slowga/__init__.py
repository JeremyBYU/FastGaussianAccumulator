import time
from pathlib import Path
from collections import namedtuple
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve

from .helper import assign_vertex_colors, create_open_3d_mesh, calc_angle_delta, plot_meshes, draw_normals
from .icosahedron import refine_icosahedron, create_icosahedron_mesh
from .projections import plot_projection, convert_lat_long, azimuth_equidistant

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

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
        self.kdtree = cKDTree(gaussian_normals, leafsize=leafsize, compact_nodes=True, balanced_tree=True)
        self.accumulator = np.zeros(self.nbuckets, dtype=np.float64)
        self.colors = np.zeros_like(gaussian_normals)
        self.gaussian_normals = gaussian_normals
        self.normals = None

    def integrate(self, normals):
        self.normals = normals
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


def filter_normals_by_phi(normals, max_phi=100, return_mask=True):
    rad = np.deg2rad(max_phi)
    # TODO this can be reduced to simple z comparison
    phi = np.arctan2(1.0, normals[:, 2])
    mask = np.abs(phi) < rad
    if return_mask:
        return normals[mask, :], mask
    else:
        return normals[mask, :]


def assign_hilbert_curve(ga: GaussianAccumulator):
    bucket_normals = ga.gaussian_normals
    bucket_normals_xy = (azimuth_equidistant(bucket_normals) * (2**16 - 1)).astype(np.uint16)
    hilbert_curve = HilbertCurve(16, 2)
    bucket_normals_hv = []
    for i in range(bucket_normals_xy.shape[0]):
        hv = hilbert_curve.distance_from_coordinates(bucket_normals_xy[i, :])
        bucket_normals_hv.append(hv)
    bucket_normals_hv = np.array(bucket_normals_hv)
    return bucket_normals_hv


def plot_hilbert_curve(ga):
    bucket_normals = ga.gaussian_normals
    colors = ga.colors
    proj = azimuth_equidistant(bucket_normals)
    bucket_normals_hv = assign_hilbert_curve(ga)
    idx_sort = np.argsort(bucket_normals_hv)
    proj = proj[idx_sort, :]
    colors = colors[idx_sort, :]
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    ax = axs[0]
    ax.scatter(proj[:, 0], proj[:, 1], c=colors)
    ax.set_title("Hilbert Curve with Azimuth Equidistant Projection")
    ax.set_xlabel("x*")
    ax.set_ylabel("y*")
    ax.plot(proj[:, 0], proj[:, 1], c='k')
    ax.axis('equal')

    ax = axs[1]
    ax.bar(np.arange(ga.nbuckets), ga.accumulator[idx_sort])
    ax.set_title("Histogram of Normal Counts sorted by Hilbert Values")
    ax.set_xlabel("Hilbert Value (Ascending)")
    ax.set_ylabel("Normal Counts")
    # ax.axis('equal')

    fig.tight_layout()
    plt.show()


def visualize_gaussian_integration(refined_icosahedron_mesh, gaussian_normals, mesh, ds=50, min_samples=10000, plot=False):
    to_integrate_normals = np.asarray(mesh.triangle_normals)
    # remove normals on bottom half of sphere
    to_integrate_normals, _ = filter_normals_by_phi(to_integrate_normals)
    # determine optimal sampling
    num_normals = to_integrate_normals.shape[0]
    ds_normals = int(num_normals / ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    # perform sampling of normals
    to_integrate_normals = to_integrate_normals[np.random.choice(
        num_normals, to_sample), :]
    # integrate normals
    gaussian_normals, mask = filter_normals_by_phi(gaussian_normals, return_mask=True)
    ga = GaussianAccumulator(gaussian_normals)
    ga.integrate(to_integrate_normals)
    ga.normalize()
    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(
        refined_icosahedron_mesh, ga.colors, mask)
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
    # plot_meshes(*meshes)
    # Show our chosen refined example
    refined_icosphere, gaussian_normals = visualize_refinement(
        ico_copy, level=3, plot=False)
    # Get an Example Mesh
    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 1:
            continue
        fname = mesh_fpath.stem
        # print(fname)
        example_mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()
        # plot_meshes(example_mesh)

        ga, colored_icosahedron = visualize_gaussian_integration(
            refined_icosphere, gaussian_normals, example_mesh)
        np.savetxt(str(REALSENSE_DIR / "{}_buckets.txt".format(fname)),
                   convert_lat_long(ga.gaussian_normals, degrees=True), fmt='%.4f')
        np.savetxt(str(REALSENSE_DIR / "{}_normals.txt".format(fname)),
                   convert_lat_long(ga.normals, degrees=True), fmt='%.4f')
        plot_hilbert_curve(ga)
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
