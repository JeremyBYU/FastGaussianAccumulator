""" Example script that visualizes the unwrapping process for peak detection 
    The steps here are much more verbose and just meant highlight the methods.
"""
import time
import functools
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d

from fastgac import GaussianAccumulatorS2Beta, MatX3d, refine_icosahedron, refine_icochart, IcoCharts
from fastgac.peak_and_cluster import find_peaks_from_ico_charts
from fastgac.o3d_util import get_colors, create_open_3d_mesh, assign_vertex_colors, plot_meshes, translate_meshes
from examples.python.run_meshes import visualize_gaussian_integration

from examples.python.util.mesh_util import get_mesh_data_iterator

def extract_chart(mesh, chart_idx=0):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    num_triangles = triangles.shape[0]
    chart_size = int(num_triangles / 5)
    chart_start_idx = chart_idx * chart_size
    chart_end_idx = chart_start_idx + chart_size
    triangles_chart = triangles[chart_start_idx:chart_end_idx, :]

    chart_mesh = create_open_3d_mesh(triangles_chart, vertices)
    chart_mesh.vertex_colors = mesh.vertex_colors

    return chart_mesh, chart_start_idx, chart_end_idx


def decompose(ico):
    triangles = np.asarray(ico.triangles)
    vertices = np.asarray(ico.vertices)
    ico_o3d = create_open_3d_mesh(triangles, vertices)
    return triangles, vertices, ico_o3d


def analyze_mesh(mesh):
    """Demonstrates unwrapping and peak detection of a S2 Histogram"""
    LEVEL = 4
    kwargs_opt_integrate = dict(num_nbr=12)

    # Create Gaussian Accumulator
    ga_cpp_s2 = GaussianAccumulatorS2Beta(level=LEVEL)
    # This function will integrate the normals and return an open3d mesh for visualization.
    colored_icosahedron_s2, _, _ = visualize_gaussian_integration(
        ga_cpp_s2, mesh, integrate_kwargs=kwargs_opt_integrate)
    num_triangles = ga_cpp_s2.num_buckets

    # for verification
    ico_s2_organized_mesh = ga_cpp_s2.copy_ico_mesh(True)
    _, _, ico_o3d_s2_om = decompose(ico_s2_organized_mesh)
    colors_s2 = get_colors(range(num_triangles), colormap=plt.cm.tab20)[:, :3]
    colored_ico_s2_organized_mesh = assign_vertex_colors(ico_o3d_s2_om, colors_s2)

    # Demonstrate the five charts for visualization
    bucket_counts = np.asarray(ga_cpp_s2.get_normalized_bucket_counts(True))
    bucket_colors = get_colors(bucket_counts)[:, :3]
    charts_triangles = []
    for chart_idx in range(5):
        chart_size = int(num_triangles / 5)
        chart_start_idx = chart_idx * chart_size
        chart_end_idx = chart_start_idx + chart_size
        icochart_square = refine_icochart(level=LEVEL, square=True)
        _, _, icochart_square_o3d = decompose(icochart_square)
        colored_icochart_square = assign_vertex_colors(
            icochart_square_o3d, bucket_colors[chart_start_idx:chart_end_idx, :])
        charts_triangles.append(colored_icochart_square)

    # Plot the unwrapped icosahedron
    new_charts = translate_meshes(charts_triangles, current_translation=-4.0, axis=1)
    all_charts = functools.reduce(lambda a, b: a + b, new_charts)
    plot_meshes(colored_ico_s2_organized_mesh, colored_icosahedron_s2, all_charts, mesh)
    avg_peaks = np.array(ga_cpp_s2.find_peaks(threshold_abs=25, cluster_distance=0.1, min_cluster_weight=0.15))
    print(avg_peaks)
    full_image = np.asarray(ga_cpp_s2.ico_chart.image)

    plt.imshow(full_image)
    plt.xticks(np.arange(0, full_image.shape[1], step=1))
    plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()


def visualize_unwrapping():
    """Demonstrate the unwrapping process by color codes sections"""
    LEVEL = 2
    ico = refine_icosahedron(level=0)
    ico_s2 = GaussianAccumulatorS2Beta(level=LEVEL)
    ico_s2_organized_mesh = ico_s2.copy_ico_mesh(True)
    triangles_ico, vertices, ico_o3d = decompose(ico)
    triangles_s2_om, _, ico_o3d_s2_om = decompose(ico_s2_organized_mesh)
    icochart_slanted = refine_icochart(level=LEVEL, square=False)
    _, _, icochart_slanted_o3d = decompose(icochart_slanted)
    icochart_square = refine_icochart(level=LEVEL, square=True)
    _, _, icochart_square_o3d = decompose(icochart_square)

    colors = get_colors(range(triangles_ico.shape[0]), colormap=plt.cm.tab20)[:, :3]
    colors_s2 = get_colors(range(triangles_s2_om.shape[0]), colormap=plt.cm.tab20)[:, :3]

    colored_ico = assign_vertex_colors(ico_o3d, colors)
    colored_ico_s2 = assign_vertex_colors(ico_o3d_s2_om, colors_s2)
    colored_icochart, start_idx, end_idx = extract_chart(colored_ico_s2, chart_idx=0)
    colored_icochart_slanted = assign_vertex_colors(icochart_slanted_o3d, colors_s2[start_idx:end_idx, :])
    colored_icochart_square = assign_vertex_colors(icochart_square_o3d, colors_s2[start_idx:end_idx, :])

    plot_meshes([colored_ico], [colored_ico_s2], colored_icochart, colored_icochart_slanted, colored_icochart_square)


def main():
    visualize_unwrapping()

    for i, mesh in enumerate(get_mesh_data_iterator()):
        if i < 0:
            continue
        analyze_mesh(mesh)


if __name__ == "__main__":
    main()
