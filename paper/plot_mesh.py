import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from fastga import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d, convert_normals_to_hilbert, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastga.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

from examples.python.run_meshes import ALL_MESHES, ALL_MESHES_ROTATIONS, visualize_gaussian_integration, plot_meshes

def main():
    EXAMPLE_INDEX = 2
    kwargs_base = dict(level=4, max_phi=180)
    kwargs_s2 = dict(**kwargs_base)
    kwargs_opt_integrate = dict(num_nbr=12)
    query_max_phi = kwargs_base['max_phi'] - 5

    # Get an Example Mesh
    ga_cpp_s2 = GaussianAccumulatorS2(**kwargs_s2)

    example_mesh = o3d.io.read_triangle_mesh(str(ALL_MESHES[EXAMPLE_INDEX]))
    r = ALL_MESHES_ROTATIONS[EXAMPLE_INDEX]
    example_mesh_filtered = example_mesh
    if r is not None:
        example_mesh_filtered = example_mesh_filtered.rotate(r.as_matrix())
        example_mesh_filtered = example_mesh_filtered.filter_smooth_laplacian(5)

    example_mesh_filtered.compute_triangle_normals()
    colored_icosahedron_s2, normals, neighbors_s2 = visualize_gaussian_integration(
        ga_cpp_s2, example_mesh_filtered, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)

    o3d.visualization.draw_geometries([example_mesh_filtered])
    o3d.visualization.draw_geometries([colored_icosahedron_s2])


    # Visualize unwrapping
    ico_chart_ = IcoCharts(4)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    find_peaks_kwargs=dict(threshold_abs=50,min_distance=1, exclude_border=False, indices=False)
    cluster_kwargs=dict(t =0.1,criterion ='distance')
    _, _, avg_peaks, _ = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex), find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs)
    print(avg_peaks)
    full_image = np.asarray(ico_chart_.image)

    plt.imshow(full_image)
    plt.axis('off')
    # plt.xticks(np.arange(0, full_image.shape[1], step=1))
    # plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()



if __name__ == "__main__":
    main()
