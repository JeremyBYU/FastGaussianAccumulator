import time
import functools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
from fastga import GaussianAccumulatorS2, MatX3d, refine_icosahedron, refine_icochart, IcoCharts
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from examples.python.run_meshes import visualize_gaussian_integration
from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors,
                               create_open_3d_mesh, assign_vertex_colors, plot_meshes, find_peaks_from_accumulator)
from src.Python.slowga.helper import translate_meshes

from skimage.feature import peak_local_max

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]



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
    LEVEL = 4
    kwargs_base = dict(level=LEVEL, max_phi=180)
    kwargs_s2 = dict(**kwargs_base)
    kwargs_opt_integrate = dict(num_nbr=12)
    query_max_phi = kwargs_base['max_phi'] - 5



    ga_cpp_s2 = GaussianAccumulatorS2(**kwargs_s2)
    colored_icosahedron_s2, _, _ = visualize_gaussian_integration(
        ga_cpp_s2, mesh, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)
    num_triangles = ga_cpp_s2.num_buckets

    # for verification
    ico_s2_organized_mesh = ga_cpp_s2.copy_ico_mesh(True)
    _, _, ico_o3d_s2_om = decompose(ico_s2_organized_mesh)
    colors_s2 = get_colors(range(num_triangles), colormap=plt.cm.tab20)[:, :3]
    colored_ico_s2_organized_mesh = assign_vertex_colors(ico_o3d_s2_om, colors_s2)

    # image = create_chart_image(ga_cpp_s2, ico_s2_organized_mesh, level=LEVEL, chart_idx=0)

    # bucket_normals = np.asarray(ga_cpp_s2.get_bucket_normals(True))
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


    new_charts = translate_meshes(charts_triangles, current_translation=-4.0, axis=1)
    all_charts = functools.reduce(lambda a,b : a+b,new_charts)
    plot_meshes(colored_ico_s2_organized_mesh, colored_icosahedron_s2, all_charts, mesh)

    ico_chart_ = IcoCharts(LEVEL)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    _, _, avg_peaks, _ = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex))
    print(avg_peaks)

    full_image = np.asarray(ico_chart_.image)

    plt.imshow(full_image)
    plt.xticks(np.arange(0, full_image.shape[1], step=1))
    plt.yticks(np.arange(0, full_image.shape[0], step=1))
    plt.show()
    # colored_image = np.ascontiguousarray(plt.cm.viridis(image)[:, :, :3], dtype=np.float32)
    


def main():
    LEVEL = 2
    ico = refine_icosahedron(level=0)
    ico_s2 = GaussianAccumulatorS2(level=LEVEL)
    ico_s2_organized_mesh = ico_s2.copy_ico_mesh(True)
    triangles_ico, vertices, ico_o3d = decompose(ico)
    triangles_s2_om, _, ico_o3d_s2_om = decompose(ico_s2_organized_mesh)
    icochart_slanted = refine_icochart(level=LEVEL, square=False)
    _, _, icochart_slanted_o3d = decompose(icochart_slanted)
    icochart_square = refine_icochart(level=LEVEL, square=True)
    _, _, icochart_square_o3d = decompose(icochart_square)

    colors = get_colors(range(triangles_ico.shape[0]), colormap=plt.cm.tab20)[:, :3]
    colors_s2 = get_colors(range(triangles_s2_om.shape[0]), colormap=plt.cm.tab20)[:, :3]
    # To verify colormapping
    # colors_s2 = np.vstack((colors_s2[::4,:], colors_s2[::4,:], colors_s2[::4,:], colors_s2[::4,:]))

    colored_ico = assign_vertex_colors(ico_o3d, colors)
    colored_ico_s2 = assign_vertex_colors(ico_o3d_s2_om, colors_s2)
    colored_icochart, start_idx, end_idx = extract_chart(colored_ico_s2, chart_idx=0)
    colored_icochart_slanted = assign_vertex_colors(icochart_slanted_o3d, colors_s2[start_idx:end_idx, :])
    colored_icochart_square = assign_vertex_colors(icochart_square_o3d, colors_s2[start_idx:end_idx, :]) 
    # colored_icochart_square_2 = assign_vertex_colors(icochart_square_o3d, get_colors(range(int(triangles_s2_om.shape[0] / 5)), colormap=plt.cm.tab20)[:, :3]) 

    plot_meshes([colored_ico], [colored_ico_s2], colored_icochart, colored_icochart_slanted, colored_icochart_square)

    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 0:
            continue
        fname = mesh_fpath.stem
        # print(fname)
        example_mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        if r is not None:
            example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()
        analyze_mesh(example_mesh)


if __name__ == "__main__":
    main()
