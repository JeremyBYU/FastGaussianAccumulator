import time
import functools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
from fastga import GaussianAccumulatorS2, MatX3d, refine_icosahedron, refine_icochart, IcoChart
from examples.python.run_meshes import visualize_gaussian_integration
from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors,
                               create_open_3d_mesh, assign_vertex_colors, plot_meshes, find_peaks_from_accumulator)
from src.Python.slowga.helper import translate_meshes

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]


def IcoMeshFaces(level = 1):
    if level == 0:
        return 20
    else:
        return IcoMeshFaces(level -1) * 4

def IcoMeshVertices(level = 1):
    if level == 1:
        return 12
    return IcoMeshVertices(level - 1) + int(1.5 * IcoMeshFaces(level - 1))

def get_chart_height(level, padding=1):
    return 2 ** (level) + (2 * padding)

# constexpr int get_chart_height(int level, int padding)
# {
#     return static_cast<int>(std::pow(2, level)) + (2 * padding);
# }

def collapse_range(row_col_idx, flattened_indices):
    row_idx = row_col_idx[0]
    col_idx = row_col_idx[1]
    for row in range(row_idx[0], row_idx[1]):
        for col in range(col_idx[0], col_idx[1]):
            flattened_indices.append([row, col])

def reduce_index_list(indices, flattened_indices):
    for row_col_idx in indices:
        collapse_range(row_col_idx, flattened_indices)

def geneate_copy_indices(level=2):
    sub_block_width = 2 ** level
    total_width = 2 ** (level + 1)
    chart_height = get_chart_height(level)
    num_charts = 5
    from_flattened_indices = []
    # Copies for first column
    to_flattened_indices = []
    to_indices = [[[i * chart_height + 1, i * chart_height + 1 + sub_block_width], [0,1]] for i in range(num_charts)]
    from_indices = [[[((i+1) % (num_charts)) * chart_height + 1, ((i+1) % (num_charts)) * chart_height + 2], [1, sub_block_width + 1]] for i in range(num_charts)]
    
    reduce_index_list(to_indices, to_flattened_indices)
    reduce_index_list(from_indices, from_flattened_indices)

    # Copies for ghost row, left block
    to_indices = [[[chart_height * (i+1) - 1, chart_height * (i+1)], [1, sub_block_width + 1]] for i in range(num_charts)]
    from_indices = [[[((i+1) % (num_charts)) * chart_height + 1, ((i+1) % (num_charts)) * chart_height + 2], [1 + sub_block_width, 1 + 2*sub_block_width]] for i in range(num_charts)]

    reduce_index_list(to_indices, to_flattened_indices)
    reduce_index_list(from_indices, from_flattened_indices)

    # Copies for ghost row, right block
    to_indices = [[[chart_height * (i+1) - 1, chart_height * (i+1)], [1 + sub_block_width, 1 + 2*sub_block_width]] for i in range(num_charts)]
    from_indices = [[[((i+1) % (num_charts)) * chart_height + 1, ((i+1) % (num_charts)) * chart_height + 1 + sub_block_width], [total_width, total_width + 1]] for i in range(num_charts)]

    reduce_index_list(to_indices, to_flattened_indices)
    reduce_index_list(from_indices, from_flattened_indices)

    print(to_flattened_indices)
    print(from_flattened_indices)
    return to_flattened_indices, from_flattened_indices



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


def create_local_to_global_point_index_map(all_triangles, chart_triangles, chart_idx, chart_num_vertices):
    chart_num_triangles = chart_triangles.shape[0]
    chart_tri_start_idx = chart_idx * chart_num_triangles
    chart_tri_end_idx = chart_tri_start_idx + chart_num_triangles

    global_chart_triangles = all_triangles[chart_tri_start_idx:chart_tri_end_idx, :]
    local_global_point_idx_map = np.zeros((chart_num_vertices, ), dtype='int')
    for tri_idx in range(chart_num_triangles):
        local_tri = chart_triangles[tri_idx,:]
        global_tri = global_chart_triangles[tri_idx, :]
        for idx in range(3):
            local_global_point_idx_map[local_tri[idx]] = global_tri[idx]
    return local_global_point_idx_map

def create_chart_image(ga, mesh, level=2, chart_idx=0):
    all_triangles = np.asarray(mesh.triangles)

    icochart_square = refine_icochart(level=level, square=True)
    chart_vertices = np.asarray(icochart_square.vertices)
    chart_triangles = np.asarray(icochart_square.triangles)

    local_global_point_idx_map = create_local_to_global_point_index_map(all_triangles, chart_triangles, chart_idx, chart_vertices.shape[0])
    
    padding = 1
    width = 2 ** (level + 1)
    height = 2 ** level
    width_padded = width + (2 * padding)
    height_padded = height + (2 * padding)

    image = np.zeros((height_padded, width_padded))

    scale = 2 ** level
    point_idx_to_image_idx = (chart_vertices * scale).astype(np.int)

    # ico_chart_ = IcoChart(level)
    # test = np.asarray(ico_chart_.local_to_global_point_idx_map)
    # print(point_idx_to_image_idx)
    # print(local_global_point_idx_map)
    # print(test)

    # test2 = np.asarray(ico_chart_.image_to_vertex_idx)
    # print(test2.shape, test2.dtype)
    # print(test2)

    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)

    for local_point_idx in range(point_idx_to_image_idx.shape[0]):
        global_point_idx =  local_global_point_idx_map[local_point_idx]
        img_coords = point_idx_to_image_idx[local_point_idx]
        img_row = img_coords[1]
        img_col = img_coords[0] + 1
        noralized_bucket_count = normalized_bucket_counts_by_vertex[global_point_idx]
        image[img_row, img_col] = noralized_bucket_count


    return image


def analyze_mesh(mesh):
    LEVEL = 2
    kwargs_base = dict(level=LEVEL, max_phi=180)
    kwargs_s2 = dict(**kwargs_base)
    kwargs_opt_integrate = dict(num_nbr=12)
    query_max_phi = kwargs_base['max_phi'] - 5

    geneate_copy_indices(LEVEL)

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


    ico_chart_ = IcoChart(LEVEL)
    # t0 = time.perf_counter()
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)
    # t1 = time.perf_counter() 
    # print(t1 - t0) # 200 microseconds for level 4

    full_image = np.asarray(ico_chart_.image)

    # plt.imshow(image)
    # plt.show()

    plt.imshow(full_image)
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
        print(example_mesh)
        analyze_mesh(example_mesh)


if __name__ == "__main__":
    main()
