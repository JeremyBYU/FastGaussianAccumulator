import time
import numpy as np
import open3d as o3d


from fastga import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d, convert_normals_to_hilbert, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastga.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals, plot_meshes
import matplotlib.pyplot as plt

from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors, create_open_3d_mesh, assign_vertex_colors)

def integrate_normals(to_integrate_normals, ga):
    to_integrate_normals_mat = MatX3d(to_integrate_normals)
    t0 = time.perf_counter()
    neighbors_idx = np.asarray(ga.integrate(to_integrate_normals_mat))
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000

    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    color_counts = get_colors(normalized_counts)[:, :3]
    # print(normalized_counts)

    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))

    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, None)
    return colored_icosahedron

    
def example_normals(normals:np.ndarray):
    LEVEL = 3
    kwargs_base = dict(level=LEVEL, max_phi=180)
    kwargs_s2 = dict(**kwargs_base)

    ga_cpp_s2 = GaussianAccumulatorS2(**kwargs_s2)
    colored_icosahedron  = integrate_normals(normals, ga_cpp_s2)
    num_triangles = ga_cpp_s2.num_buckets


    ico_chart_ = IcoCharts(LEVEL)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    find_peaks_kwargs=dict(threshold_abs=60, min_distance=1,
                            exclude_border=True, indices=False)
    cluster_kwargs=dict(t=0.10, criterion='distance')

    _, _, avg_peaks, _ = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex), find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs)
    print(avg_peaks)
    o3d.visualization.draw_geometries([colored_icosahedron])

    full_image = np.asarray(ico_chart_.image)

    plt.imshow(full_image)
    plt.xticks(np.arange(0, full_image.shape[1], step=1))
    plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()

def main():
    normals = np.loadtxt('fixtures/normals/bad_normals.txt')
    example_normals(normals)

if __name__ == "__main__":
    main()