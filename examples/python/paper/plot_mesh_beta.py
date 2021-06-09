import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from fastgac import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d, convert_normals_to_hilbert, IcoCharts, GaussianAccumulatorS2Beta
from fastgac.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastgac.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals

from examples.python.run_meshes import visualize_gaussian_integration, plot_meshes
from examples.python.util.mesh_util import ALL_MESHES, ALL_MESHES_ROTATIONS

def main():
    EXAMPLE_INDEX = 1
    kwargs_base = dict(level=4)
    kwargs_s2 = dict(**kwargs_base)
    kwargs_opt_integrate = dict(num_nbr=12)
    query_max_phi = 175

    # Get an Example Mesh
    ga_cpp_s2 = GaussianAccumulatorS2Beta(**kwargs_s2)

    example_mesh = o3d.io.read_triangle_mesh(str(ALL_MESHES[EXAMPLE_INDEX]))
    r = ALL_MESHES_ROTATIONS[EXAMPLE_INDEX]
    example_mesh_filtered = example_mesh
    if r is not None:
        example_mesh_filtered = example_mesh_filtered.rotate(r.as_matrix())
        example_mesh_filtered = example_mesh_filtered.filter_smooth_laplacian(5)

    example_mesh_filtered.compute_triangle_normals()
    # np.save('fixtures/normals/basement.npy', np.asarray(example_mesh_filtered.triangle_normals))
    colored_icosahedron_s2, normals, neighbors_s2 = visualize_gaussian_integration(
        ga_cpp_s2, example_mesh_filtered, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)


    o3d.visualization.draw_geometries([example_mesh_filtered])
    o3d.visualization.draw_geometries([colored_icosahedron_s2])


    # Visualize unwrapping
    ico_chart_ = IcoCharts(kwargs_base['level'])
    t2 = time.perf_counter()
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    average_bucket_normals = np.asarray(ga_cpp_s2.get_bucket_average_normals(True))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(average_bucket_normals))
    pcd.paint_uniform_color([1, 0, 0])
    average_vertex_normals = np.asarray(ga_cpp_s2.get_average_normals_by_vertex(True))


    find_peaks_kwargs=dict(threshold_abs=50,min_distance=1, exclude_border=False, indices=False)
    print(np.asarray(ico_chart_.image).shape)
    cluster_kwargs=dict(t =0.1,criterion ='distance')
    _, _, avg_peaks, avg_weights = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex), vertices=average_vertex_normals, find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs)
    t3 = time.perf_counter()
    print(t3 -t2)
    print(avg_peaks)
    # import ipdb; ipdb.set_trace()



    arrow_avg_peaks = get_arrow_normals(avg_peaks, avg_weights)
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(colored_icosahedron_s2)
    o3d.visualization.draw_geometries([colored_icosahedron_s2, *arrow_avg_peaks, wireframe])
    # o3d.visualization.draw_geometries([colored_icosahedron_s2, *arrow_avg_peaks, pcd])

    full_image = np.asarray(ico_chart_.image)

    plt.imshow(full_image)
    plt.axis('off')
    # plt.xticks(np.arange(0, full_image.shape[1], step=1))
    # plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()



if __name__ == "__main__":
    main()


"""Mesh
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.8764505760969685, 3.0280973667097442, 3.045776668203259 ],
			"boundingbox_min" : [ -2.2365574934452548, -3.6804227036671078, 0.51828136237409295 ],
			"field_of_view" : 60.0,
			"front" : [ -0.43966986583569911, 0.57136927624194478, 0.69298453030552898 ],
			"lookat" : [ 0.30001921841467899, -0.99779994278506134, 1.5071575255263165 ],
			"up" : [ 0.44135525764305411, -0.53453483690843095, 0.72074825333268089 ],
			"zoom" : 0.31999999999999978
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""


"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.1339119391275889, 1.1343327326857235, 1.1998729449684717 ],
			"boundingbox_min" : [ -1.1353148374296551, -1.0, -1.1999606130137823 ],
			"field_of_view" : 60.0,
			"front" : [ -0.59564118276660283, 0.48513744010499366, 0.6401978175538996 ],
			"lookat" : 
			[
				-0.00070144915103309557,
				0.067166366342861772,
				-4.3834022655286908e-05
			],
			"up" : [ 0.47207151576167344, -0.43341779039025202, 0.76765715197587236 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""