import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from fastga import GaussianAccumulatorKD, MatX3d
import matplotlib.pyplot as plt

from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors, 
                                create_open_3d_mesh, assign_vertex_colors, plot_meshes, find_peaks_from_accumulator)


THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]


def plot_hilbert_curve(ga:GaussianAccumulatorKDPy):
    proj = np.asarray(ga.get_bucket_projection())
    normals = np.asarray(ga.get_bucket_normals())
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    colors = get_colors(normalized_counts)[:,:3]
    bucket_normals_hv = np.asarray(ga.get_bucket_indices())
    # print(bucket_normals_hv)
    num_buckets = ga.num_buckets
    # print(np.max(bucket_normals_hv), np.min(bucket_normals_hv))
    idx_sort = np.argsort(bucket_normals_hv)
    proj = proj[idx_sort, :]

    accumulator_normalized_sorted = normalized_counts[idx_sort]
    gaussian_normals_sorted = normals[idx_sort, :]
    

    # Find Peaks
    peaks, clusters = find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted)

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
    ax.bar(np.arange(num_buckets), accumulator_normalized_sorted)
    ax.scatter(peaks, accumulator_normalized_sorted[peaks], marker='x', c=clusters, cmap='tab20')

    ax.set_title("Histogram of Normal Counts sorted by Hilbert Values")
    ax.set_xlabel("Hilbert Value (Ascending)")
    ax.set_ylabel("Normal Counts")

    # ax.axis('equal')
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.tight_layout()
    plt.show()
    return gaussian_normals_sorted


def visualize_gaussian_integration(ga: GaussianAccumulatorKDPy, mesh: o3d.geometry.TriangleMesh, ds=50, min_samples=10000):
    num_buckets = ga.num_buckets
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
    mask = np.asarray(ga.mask)
    query_size = to_integrate_normals.shape[0]
    # integrate normals
    if type(ga).__name__ == 'GaussianAccumulatorKD':
        to_integrate_normals = MatX3d(to_integrate_normals)
        mask = np.ma.make_mask(mask)

    t0 = time.perf_counter()
    ga.integrate(to_integrate_normals)
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000
    print("KD tree size: {}; Query Size (K): {}; Execution Time(ms): {:.1f}".format(
        num_buckets, query_size, elapsed_time))
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    color_counts = get_colors(normalized_counts)[:,:3]

    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))

    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, mask)
    return colored_icosahedron

def create_line_set(normals_sorted):
    normals_o3d = o3d.utility.Vector3dVector(normals_sorted)
    line_idx = [[i, i+1] for i  in range(normals_sorted.shape[0] - 1)]
    line_idx_o3d = o3d.utility.Vector2iVector(line_idx)
    return o3d.geometry.LineSet(normals_o3d, line_idx_o3d)

def main():
    kwargs = dict(level=4, max_phi=100, max_leaf_size=16)
    # print(gaussian_normals)
    # Get an Example Mesh
    ga_py_kdd = GaussianAccumulatorKDPy(**kwargs)
    ga_cpp_kdd = GaussianAccumulatorKD(**kwargs)
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

        colored_icosahedron_py = visualize_gaussian_integration(ga_py_kdd, example_mesh)
        colored_icosahedron_cpp = visualize_gaussian_integration(ga_cpp_kdd, example_mesh)
        # plot_projection(ga)
        # plot_hilbert_curve(ga)
        plot_meshes(colored_icosahedron_py, colored_icosahedron_cpp, example_mesh)
        # plot_hilbert_curve(ga_py_kdd)
        normals_sorted = plot_hilbert_curve(ga_cpp_kdd)
        plot_meshes([colored_icosahedron_cpp, create_line_set(normals_sorted * 1.01)])

        ga_py_kdd.clear_count()
        ga_cpp_kdd.clear_count()


if __name__ == "__main__":
    main()
