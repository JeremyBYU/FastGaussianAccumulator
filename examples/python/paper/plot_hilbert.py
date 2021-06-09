
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from fastgac import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d, convert_normals_to_hilbert, IcoCharts
from fastgac.peak_and_cluster import find_peaks_from_accumulator, find_peaks_from_ico_charts
from fastgac.o3d_util import get_arrow, get_pc_all_peaks, get_arrow_normals, create_open_3d_mesh, assign_vertex_colors, plot_meshes

from examples.python.run_meshes import create_line_set
from examples.python.util.line_mesh import LineMesh
# from src.Pyth import create_open_3d_mesh, assign_vertex_colors, plot_meshes

def main():
    kwargs_base = dict(level=4, max_phi=180)
    kwargs_s2 = dict(**kwargs_base)
    ga_cpp_s2 = GaussianAccumulatorS2(**kwargs_s2)
    normals_sorted_cube_hilbert = np.asarray(ga_cpp_s2.get_bucket_normals())

    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga_cpp_s2.mesh.triangles), np.asarray(ga_cpp_s2.mesh.vertices))

    indices = np.asarray(ga_cpp_s2.get_bucket_sfc_values())
    colors = indices / np.iinfo(indices.dtype).max
    colors = cm.viridis(colors)[:, :3]

    lm = LineMesh(normals_sorted_cube_hilbert * 1.01, colors=colors, radius=0.004)

    # colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, colors)
    # plot_meshes([refined_icosahedron_mesh, create_line_set(normals_sorted_cube_hilbert * 1.01)])
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(refined_icosahedron_mesh)
    refined_icosahedron_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    plot_meshes([refined_icosahedron_mesh, ls, *lm.cylinder_segments])


if __name__ == "__main__":
    main()


# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 1.7098574459629279, 0.70987207591299273, 1.0130857360820043 ],
# 			"boundingbox_min" : [ 0.29014255403707201, -1.0130857026947551, -1.0130857360820043 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.0040044546059786173, 0.11016896364951138, 0.99390480620213395 ],
# 			"lookat" : [ 1.0, -0.15160681339088117, 0.0 ],
# 			"up" : [ 0.0019505060267622787, 0.99391174507838853, -0.11016187417374566 ],
# 			"zoom" : 0.69999999999999996
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }