import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from fastga import GaussianAccumulatorKD, GaussianAccumulatorOpt, GaussianAccumulatorS2, MatX3d, convert_normals_to_hilbert
import matplotlib.pyplot as plt

from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors, 
                                create_open_3d_mesh, assign_vertex_colors, plot_meshes, find_peaks_from_accumulator)


THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]


def plot_hilbert_curve(ga:GaussianAccumulatorKDPy):
    proj = np.asarray(ga.get_bucket_projection())
    normals = np.asarray(ga.get_bucket_normals())
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    colors = get_colors(normalized_counts)[:,:3]
    bucket_normals_hv = np.asarray(ga.get_bucket_indices())

    # print(bucket_normals_hv)
    num_buckets = ga.num_buckets
    # print(np.max(bucket_normals_hv), np.min(bucket_normals_hv))
    idx_sort = np.argsort(bucket_normals_hv)
    bucket_normals_hv_sorted = bucket_normals_hv[idx_sort]
    proj = proj[idx_sort, :]

    # print(bucket_normals_hv)
    # print(bucket_normals_hv_sorted)

    accumulator_normalized_sorted = normalized_counts[idx_sort]
    gaussian_normals_sorted = normals[idx_sort, :]
    

    # Find Peaks
    peaks, clusters = find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted)

    colors = colors[idx_sort, :]
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    ax = axs[0]
    scatter1 = ax.scatter(proj[:, 0], proj[:, 1], c=colors, label='Projected Buckets')
    scatter2 = ax.scatter(proj[peaks, :][:, 0], proj[peaks, :][:, 1], marker='x', c=clusters, label='Clusters', cmap='tab20')
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
    ax.bar(np.arange(num_buckets), accumulator_normalized_sorted)
    ax.scatter(peaks, accumulator_normalized_sorted[peaks], marker='x', c=clusters, cmap='tab20')

    ax.set_title("Histogram of Normal Counts sorted by Hilbert Values")
    ax.set_xlabel("Hilbert Value (Ascending)")
    ax.set_ylabel("Normal Counts")

    # ax.axis('equal')
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.tight_layout()
    plt.show()
    return gaussian_normals_sorted


def visualize_gaussian_integration(ga: GaussianAccumulatorKDPy, mesh: o3d.geometry.TriangleMesh, ds=50, min_samples=10000, max_phi=100, integrate_kwargs=dict()):
    num_buckets = ga.num_buckets
    to_integrate_normals = np.asarray(mesh.triangle_normals)
    # remove normals on bottom half of sphere
    to_integrate_normals, _ = filter_normals_by_phi(to_integrate_normals, max_phi=max_phi)
    # determine optimal sampling
    num_normals = to_integrate_normals.shape[0]
    ds_normals = int(num_normals / ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    ds_step = int(num_normals / to_sample)
    # perform sampling of normals
    to_integrate_normals = to_integrate_normals[0:num_normals:ds_step, :]
    mask = np.asarray(ga.mask)
    query_size = to_integrate_normals.shape[0]

    mask = np.ones((np.asarray(ga.mesh.triangles).shape[0],), dtype=bool)
    mask[num_buckets:] = False
    class_name_str = type(ga).__name__
    # integrate normals
    if class_name_str in ['GaussianAccumulatorKD', 'GaussianAccumulatorOpt', 'GaussianAccumulatorS2']:
        to_integrate_normals = MatX3d(to_integrate_normals)
        # mask = np.ma.make_mask(mask)

    # triangles = np.asarray(ga.mesh.triangles)
    t0 = time.perf_counter()
    neighbors_idx = np.asarray(ga.integrate(to_integrate_normals, **integrate_kwargs))
    t1 = time.perf_counter()
    elapsed_time = (t1 - t0) * 1000
    print("{}; KD tree size: {}; Query Size (K): {}; Execution Time(ms): {:.1f}".format(
        class_name_str, num_buckets, query_size, elapsed_time))
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    color_counts = get_colors(normalized_counts)[:,:3]
    # print(normalized_counts)

    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))

    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, mask)
    return colored_icosahedron, np.asarray(to_integrate_normals), neighbors_idx

def create_line_set(normals_sorted):
    normals_o3d = o3d.utility.Vector3dVector(normals_sorted)
    line_idx = [[i, i+1] for i  in range(normals_sorted.shape[0] - 1)]
    line_idx_o3d = o3d.utility.Vector2iVector(line_idx)
    return o3d.geometry.LineSet(normals_o3d, line_idx_o3d)

def plot_issues_2(idx, normals, chosen_buckets, ga, mesh):

    normal_idx = idx[0]
    bad_normals = normals[idx,:]
    bad_chosen_triangles = chosen_buckets[idx]

    chosen_triangle = bad_chosen_triangles[0]
    bad_normal = np.expand_dims(bad_normals[0, :], axis=0)
    print(bad_normal)
    # chosen_normal = 

    # Get all bucket normals
    bucket_normals_sorted = np.asarray(ga.get_bucket_normals())

    # Get 2D projection of bad normal
    projected_normals, hv = convert_normals_to_hilbert(MatX3d(normals), ga.projected_bbox)
    projected_normals = np.asarray(projected_normals)
    bad_projected_normal = projected_normals[normal_idx, :]

    # Get Projected coordinates of buckets
    proj = np.asarray(ga.get_bucket_projection())
    normalized_counts = np.asarray(ga.get_normalized_bucket_counts())
    colors = get_colors(normalized_counts)[:,:3]


    # Get Projected Coordinates of Buckets and normals!
    all_normals = np.vstack((bucket_normals_sorted, normals))
    all_projected_normals, all_hv = convert_normals_to_hilbert(MatX3d(all_normals), ga.projected_bbox)
    all_projected_normals = np.asarray(all_projected_normals)
    all_hv = np.asarray(all_hv)

    idx_sort = np.argsort(all_hv)
    all_projected_normals = all_projected_normals[idx_sort]
    all_normals_sorted = np.ascontiguousarray(all_normals[idx_sort])

    

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    ax = axs
    # plot buckets
    scatter1 = ax.scatter(proj[:, 0], proj[:, 1], c=colors, label='Projected Buckets')
    scatter2 = ax.scatter(proj[chosen_triangle, 0], proj[chosen_triangle, 1], c='r', label='Chosen Triangle')
    # scatter3 = ax.scatter(proj[133, 0], proj[133, 1], c='g', label='Hilbert Mapped Triangle')
    scatter4 = ax.scatter(bad_projected_normal[0], bad_projected_normal[1], c=[[0.5, 0.5, 0.5]], label='Projected Normal')
    line1 = ax.plot(all_projected_normals[:, 0], all_projected_normals[:, 1], c='k', label='Hilbert Curve Connections')[0]
    leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
    # scatter2 = ax.scatter(proj[133, 0], proj[133, 1], c='k')
    plt.show()


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bad_normal)
    pcd.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(bucket_normals_sorted)
    pcd2.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0]])

    vertex_colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    p_idx = triangles[chosen_triangle,:]
    vertex_colors[p_idx] = [1, 0, 0]
    # p_idx = triangles[133,:]
    # vertex_colors[p_idx] = [0, 1, 0]

    ls = create_line_set(all_normals_sorted * 1.002)
    o3d.visualization.draw_geometries([mesh, pcd, pcd2, ls])

def reorder_single_hv_to_s2(ga_cpp_kdd, ga_cpp_s2):
    s2_normals = ga_cpp_s2.get_bucket_normals()
    _, hv = convert_normals_to_hilbert(s2_normals, ga_cpp_kdd.projected_bbox)
    hv = np.asarray(hv)
    idx_sort = np.argsort(hv)
    hv = hv[idx_sort]
    return idx_sort

def main():
    kwargs_base = dict(level=4, max_phi=150)
    kwargs_kdd = dict(**kwargs_base, max_leaf_size=10)
    kwargs_opt = dict(**kwargs_base)
    kwargs_s2 = dict(**kwargs_base)

    kwargs_opt_integrate = dict(num_nbr=12)
    # Get an Example Mesh
    ga_py_kdd = GaussianAccumulatorKDPy(**kwargs_kdd)
    ga_cpp_kdd = GaussianAccumulatorKD(**kwargs_kdd)
    ga_cpp_opt = GaussianAccumulatorOpt(**kwargs_opt)
    ga_cpp_s2 = GaussianAccumulatorS2(**kwargs_s2)

    query_max_phi = kwargs_base['max_phi'] - 5

    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 1:
            continue
        fname = mesh_fpath.stem
        # print(fname)
        example_mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        if r is not None:
            example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()

        colored_icosahedron_py, normals, neighbors_py = visualize_gaussian_integration(ga_py_kdd, example_mesh, max_phi=query_max_phi)
        colored_icosahedron_cpp, normals, neighbors_cpp = visualize_gaussian_integration(ga_cpp_kdd, example_mesh, max_phi=query_max_phi)
        colored_icosahedron_opt, normals, neighbors_opt = visualize_gaussian_integration(ga_cpp_opt, example_mesh, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)
        colored_icosahedron_s2, normals, neighbors_s2 = visualize_gaussian_integration(ga_cpp_s2, example_mesh, max_phi=query_max_phi, integrate_kwargs=kwargs_opt_integrate)

        idx_opt, = np.nonzero(neighbors_opt.astype(np.int64) - neighbors_cpp.astype(np.int64))
        print("Fast GaussianAccumulatorOpt (Hemisphere) incorrectly assigned : {}".format(idx_opt.shape[0]))

        reorder_s2 = reorder_single_hv_to_s2(ga_cpp_kdd, ga_cpp_s2)
        idx_s2, = np.nonzero(reorder_s2[neighbors_cpp].astype(np.int64) - neighbors_s2.astype(np.int64))
        print("Fast GaussianAccumulatorS2 (Full Sphere) incorrectly assigned : {}".format(idx_s2.shape[0])) # Doesn't work because sorting is different

        if idx_opt.shape[0] > 0:
            pass
            # plot_issues_2(idx_opt, normals, neighbors_opt, ga_cpp_opt, colored_icosahedron_opt)

        plot_meshes(colored_icosahedron_py, colored_icosahedron_cpp, colored_icosahedron_opt, colored_icosahedron_s2, example_mesh)
        # plot_hilbert_curve(ga_py_kdd)
        # normals_sorted = plot_hilbert_curve(ga_cpp_kdd)
        plot_hilbert_curve(ga_cpp_opt)
        normals_sorted_proj_hilbert = np.asarray(ga_cpp_opt.get_bucket_normals())
        normals_sorted_cube_hilbert = np.asarray(ga_cpp_s2.get_bucket_normals())
        plot_meshes([colored_icosahedron_cpp, create_line_set(normals_sorted_proj_hilbert * 1.01)], [colored_icosahedron_s2, create_line_set(normals_sorted_cube_hilbert * 1.01)])

        ga_py_kdd.clear_count()
        ga_cpp_kdd.clear_count()
        ga_cpp_opt.clear_count()
        ga_cpp_s2.clear_count()


if __name__ == "__main__":
    main()
