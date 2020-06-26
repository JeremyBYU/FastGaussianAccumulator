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
from .icosahedron import create_icosahedron, refine_icosahedron, create_icosahedron_mesh
from .projections import plot_projection, convert_lat_long, azimuth_equidistant
from .peak_and_cluster import find_peaks_from_accumulator

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]

def get_colors(inp, colormap=plt.cm.viridis, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def refined_ico_mesh(level=0):
    triangles, vertices = create_icosahedron()
    vertices, triangles = refine_icosahedron(triangles, vertices, level=level)
    new_mesh = create_open_3d_mesh(triangles, vertices)
    return np.array(new_mesh.triangle_normals), new_mesh


class GaussianAccumulatorKDPy(object):
    def __init__(self, level=0, max_phi=100, max_leaf_size=16):
        super().__init__()
        bucket_normals, mesh = refined_ico_mesh(level=level)
        bucket_normals, mask = filter_normals_by_phi(bucket_normals, max_phi)
        self.bucket_normals = bucket_normals
        self.bucket_indices = assign_hilbert_curve(bucket_normals)
        self.mask = mask
        self.mesh = mesh
        self.num_buckets = bucket_normals.shape[0]
        self.accumulator = np.zeros(self.num_buckets, dtype=np.int32)
        self.accumulator_normalized = np.zeros(self.num_buckets, dtype=np.float64)
        self.bucket_projection = azimuth_equidistant(bucket_normals)

        # Sort buckets by hilbert indices
        idx_sort = np.argsort(self.bucket_indices)
        self.bucket_indices  = np.ascontiguousarray(self.bucket_indices[idx_sort])
        self.bucket_projection  = np.ascontiguousarray(self.bucket_projection[idx_sort])
        self.bucket_normals  = np.ascontiguousarray(self.bucket_normals[idx_sort])
        # fix triangles
        triangles = np.asarray(self.mesh.triangles)
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.vstack((triangles[mask][idx_sort], triangles[~mask]))
        self.mesh = create_open_3d_mesh(triangles, vertices)
        self.kdtree = cKDTree(self.bucket_normals, leafsize=max_leaf_size, compact_nodes=True, balanced_tree=True)

    def integrate(self, normals):
        _, neighbors = self.kdtree.query(normals)
        np.add.at(self.accumulator, neighbors, 1)
        return neighbors
        # for idx in neighbors:
        #     self.accumulator[idx] = self.accumulator[idx] + 1
    
    def get_bucket_normals(self):
        return self.bucket_normals

    def get_normalized_bucket_counts(self):
        # print(np.max(self.accumulator))
        self.accumulator_normalized = self.accumulator / np.max(self.accumulator)
        return self.accumulator_normalized

    def get_bucket_sfc_values(self):
        return self.bucket_indices

    def get_bucket_projection(self):
        return self.bucket_projection

    def clear_count(self):
        self.accumulator.fill(0)


class GaussianAccumulator(object):
    def __init__(self, gaussian_normals, leafsize=16):
        super().__init__()
        self.nbuckets = gaussian_normals.shape[0]
        self.kdtree = cKDTree(gaussian_normals, leafsize=leafsize, compact_nodes=True, balanced_tree=True)
        self.accumulator = np.zeros(self.nbuckets, dtype=np.int32)
        self.accumulator_normalized = np.zeros(self.nbuckets, dtype=np.float64)
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
        self.accumulator_normalized = self.accumulator / np.max(self.accumulator)
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
    # print(vertices, vertices.shape)
    # print(triangles, triangles.shape)
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
    min_z = np.cos(rad)
    # phi = np.arctan2(1.0, normals[:, 2])
    # mask = np.abs(phi) < rad
    mask = normals[:, 2] >= min_z
    if return_mask:
        return normals[mask, :], mask
    else:
        return normals[mask, :]


# def assign_hilbert_curve(ga: GaussianAccumulator):
#     dtype = np.uint32
#     max_length_axis = 2**16 - 1
#     bucket_normals = ga.gaussian_normals
#     bucket_normals_xy = (azimuth_equidistant(bucket_normals) * max_length_axis).astype(dtype)
#     hilbert_curve = HilbertCurve(16, 2)
#     bucket_normals_hv = []
#     for i in range(bucket_normals_xy.shape[0]):
#         hv = hilbert_curve.distance_from_coordinates(bucket_normals_xy[i, :])
#         bucket_normals_hv.append(hv)
#     bucket_normals_hv = np.array(bucket_normals_hv)
#     return bucket_normals_hv

def assign_hilbert_curve(normals:np.ndarray):
    dtype = np.uint32
    max_length_axis = 2**16 - 1
    bucket_normals = normals
    bucket_normals_xy = (azimuth_equidistant(bucket_normals) * max_length_axis).astype(dtype)
    # print(bucket_normals_xy)
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
    bucket_normals_hv = assign_hilbert_curve(ga.gaussian_normals)
    # print(np.max(bucket_normals_hv), np.min(bucket_normals_hv))
    idx_sort = np.argsort(bucket_normals_hv)
    proj = proj[idx_sort, :]
    accumulator_normalized_sorted = ga.accumulator_normalized[idx_sort]
    gaussian_normals_sorted = ga.gaussian_normals[idx_sort]

    # Find Peaks
    peaks, clusters, _, _ = find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted)

    colors = colors[idx_sort, :]
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    ax = axs[0]
    scatter1 = ax.scatter(proj[:, 0], proj[:, 1], c=colors, label='Projected Buckets')
    scatter2 = ax.scatter(proj[peaks, :][:, 0], proj[peaks, :][:, 1], marker='x', c=clusters, label='Clusters', cmap='tab20')
    ax.set_title("Hilbert Curve with Azimuth Equidistant Projection")
    ax.set_xlabel("x*")
    ax.set_ylabel("y*")
    line1 = ax.plot(proj[:, 0], proj[:, 1], c='k', label='Hilbert Curve Connections')[0]
    ax.axis('equal')
    leg = ax.legend(loc='upper left', fancybox=True, shadow=True)

    # we will set up a dict mapping legend line to orig line, and enable
    # picking on the legend line
    lines = [line1, scatter1, scatter2]
    lined = dict()
    for legline, origline in zip(leg.legendHandles, lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline

    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    ax = axs[1]
    ax.bar(np.arange(ga.nbuckets), accumulator_normalized_sorted)
    ax.scatter(peaks, accumulator_normalized_sorted[peaks], marker='x', c=clusters, cmap='tab20')

    ax.set_title("Histogram of Normal Counts sorted by Hilbert Values")
    ax.set_xlabel("Hilbert Value (Ascending)")
    ax.set_ylabel("Normal Counts")

    # ax.axis('equal')
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.tight_layout()
    plt.show()

# def find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted,
#                find_peaks_kwargs=dict(height=0.10, threshold=None, distance=4, width=None, prominence=None),
#                cluster_kwargs=dict(t=0.15, criterion='distance')):
#     t0 = time.perf_counter()
#     peaks, _ = find_peaks(accumulator_normalized_sorted, **find_peaks_kwargs)
#     t1 = time.perf_counter()

#     gaussian_normal_1d_clusters = gaussian_normals_sorted[peaks,:]
#     Z = linkage(gaussian_normal_1d_clusters, 'single')
#     clusters = fcluster(Z, **cluster_kwargs)
#     t2 = time.perf_counter()

#     print("Peak Detection - Find Peaks Execution Time (ms): {:.1f}; Hierarchical Clustering Execution Time (ms): {:.1f}".format((t1-t0) * 1000, (t2-t1) * 1000))
#     return peaks, clusters


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
    family = [1, 2, 3, 4]
    meshes = generate_family_of_icosahedron(np.asarray(
        ico.triangles), np.asarray(ico.vertices), family)
    family.insert(0, 0)
    meshes.insert(0, ico)
    for level, mesh in zip(family, meshes):
        angle_diff = calc_angle_delta(mesh, level)
        print("Refinement Level: {}; Number of Vertices: {}, Number of Triangles: {}, Angle Difference: {:.1f}".format(
            level, np.array(mesh.vertices).shape[0], np.array(mesh.triangles).shape[0], angle_diff))
    # meshes.insert(0, sphere)
    plot_meshes(*meshes)
    return
    # Show our chosen refined example
    refined_icosphere, gaussian_normals = visualize_refinement(
        ico_copy, level=4, plot=False)
    # print(gaussian_normals)
    # Get an Example Mesh
    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 0:
            continue
        fname = mesh_fpath.stem
        # print(fname)
        example_mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        if r is not None:
            example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()
        # plot_meshes(example_mesh)

        ga, colored_icosahedron = visualize_gaussian_integration(
            refined_icosphere, gaussian_normals, example_mesh)
        # np.savetxt(str(REALSENSE_DIR / "{}_buckets.txt".format(fname)),
        #            convert_lat_long(ga.gaussian_normals, degrees=True), fmt='%.4f')
        # np.savetxt(str(REALSENSE_DIR / "{}_normals.txt".format(fname)),
        #            convert_lat_long(ga.normals, degrees=True), fmt='%.4f')
        plot_projection(ga)
        plot_hilbert_curve(ga)
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
