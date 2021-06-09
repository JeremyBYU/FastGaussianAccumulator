""" Basic demonstation of integrating normals and performing peak detection

"""
import time
import sys
import argparse
import numpy as np
import open3d as o3d

from fastgac import GaussianAccumulatorS2Beta, MatX3d
from fastgac.o3d_util import assign_vertex_colors, plot_meshes, get_colors, create_open_3d_mesh
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
    kwargs_s2 = dict(level=4)

    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.5).translate([-2.0, 0, 0])
    # Create Gaussian Accumulator
    ga_cpp_s2 = GaussianAccumulatorS2Beta(**kwargs_s2)
    # Integrate the normals and get open3d visualization
    colored_icosahedron  = integrate_normals_and_visualize(normals, ga_cpp_s2)
    o3d.visualization.draw_geometries([colored_icosahedron, axis_frame])
    # New simplified API for finding peaks
    res = np.array(ga_cpp_s2.find_peaks(threshold_abs=20, cluster_distance=0.1, min_cluster_weight=0.2))
    print("New Detected Peaks:")
    res = sort_by_distance_from_point(res)
    print(res)


    full_image = np.asarray(ga_cpp_s2.ico_chart.image)
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
    example_normals(combined)

if __name__ == "__main__":
    main()