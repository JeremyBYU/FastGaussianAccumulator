""" Basic demonstation of integrating normals and performing peak detection

"""
import time
import sys
import argparse
import numpy as np
import open3d as o3d

from fastgac import GaussianAccumulatorS2Beta, MatX3d, convert_normals_to_hilbert, IcoCharts
from fastgac.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastgac.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals, plot_meshes, assign_vertex_colors, plot_meshes, get_colors, create_open_3d_mesh
import matplotlib.pyplot as plt

from tests.python.helpers.setup_helper import cluster_normals, sort_by_distance_from_point
np.random.seed(1)

np.set_printoptions(suppress=True, precision=3)

def integrate_normals_and_visualize(to_integrate_normals, ga):
    to_integrate_normals_mat = MatX3d(to_integrate_normals)
    t0 = time.perf_counter()
    neighbors_idx = np.asarray(ga.integrate(to_integrate_normals_mat))
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    color_counts = get_colors(normalized_counts)[:, :3]
    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, None)

    return colored_icosahedron

    
def example_normals(normals:np.ndarray):
    LEVEL = 4
    kwargs_base = dict(level=LEVEL)
    kwargs_s2 = dict(**kwargs_base)

    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5).translate([-2.0, 0, 0])
    # Create Gaussian Accumulator
    ga_cpp_s2 = GaussianAccumulatorS2Beta(**kwargs_s2)
    # Integrate the normals and get open3d visualization
    colored_icosahedron  = integrate_normals_and_visualize(normals, ga_cpp_s2)
    o3d.visualization.draw_geometries([colored_icosahedron, axis_frame])
    # Create the IcoChart for unwrapping
    ico_chart_ = IcoCharts(LEVEL)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)
    average_vertex_normals = np.asarray(ga_cpp_s2.get_average_normals_by_vertex(True))

    # 2D Peak Detection
    find_peaks_kwargs = dict(threshold_abs=20, min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs = dict(t=0.05, criterion='distance')
    average_filter = dict(min_total_weight=0.2)

    # New simplified API for finding peaks
    res = np.array(ga_cpp_s2.find_peaks(threshold_abs=find_peaks_kwargs['threshold_abs'], cluster_distance=cluster_kwargs['t'], min_cluster_weight=average_filter['min_total_weight']))
    print("New Detected Peaks:")
    res = sort_by_distance_from_point(res)
    print(res)

    # Old Way of finding peaks
    _, _, avg_peaks, _ = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex), vertices=average_vertex_normals, find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs)
    avg_peaks = sort_by_distance_from_point(avg_peaks)
    print("Detected Peaks:")
    print(avg_peaks)

    full_image = np.asarray(ico_chart_.image)
    plt.imshow(full_image)
    plt.xticks(np.arange(0, full_image.shape[1], step=1))
    plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()

    # Don't forget to reset the GA
    ga_cpp_s2.clear_count()

def main():
    parser = argparse.ArgumentParser(description='Integrate some Normals')
    parser.add_argument('--path', type=str, help='Specify an optional file')
    args = parser.parse_args()
    if args.path is None:
        clusters, normals = cluster_normals(10, 1000, patch_deg=5)
        combined =np.concatenate(clusters)
    else:
        print("loading data from ", args.path)
        data = np.load(args.path)
        combined = data['clusters']
        normals = data['normals']
    print(sort_by_distance_from_point(normals))
    # sys.exit()
    # normals = np.asarray([
    #     [0.0, 0.0, 0.95],
    #     [0.0, 0.0, 0.98],
    #     [0.95, 0.0, 0],
    #     [0.98, 0.0, 0]
    # ])
    example_normals(combined)

if __name__ == "__main__":
    main()