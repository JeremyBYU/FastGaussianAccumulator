""" This example demonstrates four (4) different forms of GaussianAccumulators. It also demonstrates two forms of peak detection.
"""
import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from fastgac import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2Beta, MatX3d, convert_normals_to_hilbert, IcoCharts
from fastgac.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastgac.helper import down_sample_normals
from fastgac.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals, plot_meshes, assign_vertex_colors, create_open_3d_mesh, get_colors, create_line_set
from examples.python.util.mesh_util import get_mesh_data_iterator

from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi)


def get_image_peaks(ga_cpp_s2, level=2, **kwargs):
    ico_chart = IcoCharts(level)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart.fill_image(normalized_bucket_counts_by_vertex)

    find_peaks_kwargs = dict(threshold_abs=20, min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs = dict(t=0.2, criterion='distance')
    average_filter = dict(min_total_weight=0.05)

    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_ico_charts(ico_chart, np.asarray(
        normalized_bucket_counts_by_vertex), find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs, average_filter=average_filter)
    gaussian_normals_sorted = np.asarray(ico_chart.sphere_mesh.vertices)
    pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
    arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)

    print(avg_peaks)

    return [pcd_all_peaks, *arrow_avg_peaks]


def plot_hilbert_curve(ga: GaussianAccumulatorKDPy, plot=False):

    normals = np.asarray(ga.get_bucket_normals())
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    colors = get_colors(normalized_counts)[:, :3]
    bucket_normals_hv = np.asarray(ga.get_bucket_sfc_values())
    num_buckets = ga.num_buckets
    idx_sort = np.argsort(bucket_normals_hv)
    bucket_normals_hv_sorted = bucket_normals_hv[idx_sort]
    colors = colors[idx_sort, :]
    accumulator_normalized_sorted = normalized_counts[idx_sort]
    gaussian_normals_sorted = normals[idx_sort, :]

    # Find Peaks using 1D signal detector
    peaks, clusters, avg_peaks, avg_weights = find_peaks_from_accumulator(
        gaussian_normals_sorted, accumulator_normalized_sorted)

    # 2D Plots
    if plot:
        class_name_str = type(ga).__name__
        if class_name_str == 'GaussianAccumulatorS2':
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        else:
            proj_ = np.asarray(ga.get_bucket_projection())
            proj = proj_[idx_sort, :]
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            ax = axs[0]
            scatter1 = ax.scatter(proj[:, 0], proj[:, 1], c=colors, label='Projected Buckets')
            scatter2 = ax.scatter(proj[peaks, :][:, 0], proj[peaks, :][:, 1],
                                  marker='x', c=clusters, label='Clusters', cmap='tab20')
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
            fig.canvas.mpl_connect('pick_event', onpick)

        ax.bar(np.arange(num_buckets), accumulator_normalized_sorted)
        ax.scatter(peaks, accumulator_normalized_sorted[peaks], marker='x', c=clusters, cmap='tab20')

        ax.set_title("Histogram of Normal Counts sorted by Hilbert Values")
        ax.set_xlabel("Hilbert Value (Ascending)")
        ax.set_ylabel("Normal Counts")
        fig.tight_layout()
        plt.show()

    pcd_all_peaks = get_pc_all_peaks(peaks, clusters, gaussian_normals_sorted)
    arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)
    return [pcd_all_peaks, *arrow_avg_peaks]


def visualize_gaussian_integration(ga: GaussianAccumulatorKDPy, mesh: o3d.geometry.TriangleMesh, integrate_kwargs=dict(), **kwargs):
    num_buckets = ga.num_buckets
    to_integrate_normals = down_sample_normals(np.asarray(mesh.triangle_normals))
    num_normals = to_integrate_normals.shape[0]

    class_name_str = type(ga).__name__
    # integrate normals
    if class_name_str in ['GaussianAccumulatorKD', 'GaussianAccumulatorOpt', 'GaussianAccumulatorS2', 'GaussianAccumulatorS2Beta']:
        to_integrate_normals = MatX3d(to_integrate_normals)

    t0 = time.perf_counter()
    neighbors_idx = np.asarray(ga.integrate(to_integrate_normals, **integrate_kwargs))
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000
    print("{}; Number of cell in GA: {}; Query Size (K): {}; Execution Time(ms): {:.1f}".format(
        class_name_str, num_buckets, num_normals, elapsed_time))

    # For visualization
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    color_counts = get_colors(normalized_counts)[:, :3]
    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, None)

    return colored_icosahedron, np.asarray(to_integrate_normals), neighbors_idx


def main():
    print("Here we are going to try out 4 different types of Gaussian Accumulators")
    print("GaussianAccumulatorKDPy = GA using k-d tree implemented in scipy")
    print("GaussianAccumulatorKD = GA using k-d tree implemented in C++ using nanoflann")
    print("GaussianAccumulatorOpt = GA spacing filling curves and local search. Optimized for top hemisphere. Don't use.")
    print("GaussianAccumulatorS2 = GA spacing filling curves and local search. Works on full sphere. This is the really the best")
    print("")
    kwargs_base = dict(level=4, max_phi=180)
    kwargs_kdd = dict(**kwargs_base, max_leaf_size=10)
    kwargs_opt = dict(**kwargs_base)
    kwargs_s2 = dict(**kwargs_base)

    kwargs_opt_integrate = dict(num_nbr=12)
    # Get an Example Mesh
    ga_py_kdd = GaussianAccumulatorKDPy(**kwargs_kdd)
    ga_cpp_kdd = GaussianAccumulatorKD(**kwargs_kdd)
    ga_cpp_opt = GaussianAccumulatorOpt(**kwargs_opt)
    ga_cpp_s2 = GaussianAccumulatorS2Beta(level=4)

    query_max_phi = kwargs_base['max_phi']

    for i, mesh in enumerate(get_mesh_data_iterator()):
        if i < 0:
            continue

        colored_icosahedron_py, normals, neighbors_py = visualize_gaussian_integration(
            ga_py_kdd, mesh, max_phi=query_max_phi)
        colored_icosahedron_cpp, normals, neighbors_cpp = visualize_gaussian_integration(
            ga_cpp_kdd, mesh, max_phi=query_max_phi)
        colored_icosahedron_opt, normals, neighbors_opt = visualize_gaussian_integration(
            ga_cpp_opt, mesh, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)
        colored_icosahedron_s2, normals, neighbors_s2 = visualize_gaussian_integration(
            ga_cpp_s2, mesh, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)

        print("Visualing the mesh and the colorized Gaussian Accumulator of type 'GaussianAccumulatorS2'")
        plot_meshes(colored_icosahedron_s2, mesh)

        # 1D Peak detection
        pcd_cpp_s2 = plot_hilbert_curve(ga_cpp_s2, plot=False)
        # 2D Peak Detection
        pcd_cpp_s2_image = get_image_peaks(ga_cpp_s2, **kwargs_base)

        normals_sorted_proj_hilbert = np.asarray(ga_cpp_opt.get_bucket_normals())
        normals_sorted_cube_hilbert = np.asarray(ga_cpp_s2.get_bucket_normals())

        print("Visualize 1D Peak Detection (Left) and 2D Peak Detection (Right).\n")
        plot_meshes([colored_icosahedron_s2, create_line_set(normals_sorted_cube_hilbert * 1.01), *pcd_cpp_s2],
                    [colored_icosahedron_s2, create_line_set(normals_sorted_cube_hilbert * 1.01), *pcd_cpp_s2_image])

        ga_py_kdd.clear_count()
        ga_cpp_kdd.clear_count()
        ga_cpp_opt.clear_count()
        ga_cpp_s2.clear_count()


if __name__ == "__main__":
    main()
