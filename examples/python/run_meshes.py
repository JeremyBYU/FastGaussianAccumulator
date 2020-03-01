import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from src.Python.slowga import GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors, create_open_3d_mesh, assign_vertex_colors, plot_meshes


THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "../../fixtures/"
REALSENSE_DIR = (FIXTURES_DIR / "realsense").absolute()
EXAMPLE_MESH_1 = REALSENSE_DIR / "example_mesh.ply"
EXAMPLE_MESH_2 = REALSENSE_DIR / "dense_first_floor_map.ply"
EXAMPLE_MESH_3 = REALSENSE_DIR / "sparse_basement.ply"

ALL_MESHES = [EXAMPLE_MESH_1, EXAMPLE_MESH_2, EXAMPLE_MESH_3]
ALL_MESHES_ROTATIONS = [None, R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0])),
                        R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))]


def visualize_gaussian_integration(ga: GaussianAccumulatorKDPy, mesh: o3d.geometry.TriangleMesh, ds=50, min_samples=10000):
    to_integrate_normals = np.asarray(mesh.triangle_normals)
    # remove normals on bottom half of sphere
    to_integrate_normals, _ = filter_normals_by_phi(to_integrate_normals)
    # determine optimal sampling
    num_normals = to_integrate_normals.shape[0]
    ds_normals = int(num_normals / ds)
    to_sample = max(min([num_normals, min_samples]), ds_normals)
    # perform sampling of normals
    to_integrate_normals = to_integrate_normals[np.random.choice(
        num_normals, to_sample), :]
    # integrate normals
    ga.integrate(to_integrate_normals)
    normalized_counts = ga.get_normalized_bucket_counts()
    color_counts = get_colors(normalized_counts)[:,:3]

    refined_icosahedron_mesh = create_open_3d_mesh(np.asarray(ga.mesh.triangles), np.asarray(ga.mesh.vertices))
    # Colorize normal buckets
    colored_icosahedron = assign_vertex_colors(refined_icosahedron_mesh, color_counts, ga.mask)
    return ga, colored_icosahedron


def main():

    ga_py = GaussianAccumulatorKDPy(level=4, max_phi=100, max_leaf_size=16)
    # print(gaussian_normals)
    # Get an Example Mesh
    for i, (mesh_fpath, r) in enumerate(zip(ALL_MESHES, ALL_MESHES_ROTATIONS)):
        if i < 0:
            continue
        fname = mesh_fpath.stem
        # print(fname)
        example_mesh = o3d.io.read_triangle_mesh(str(mesh_fpath))
        if r is not None:
            example_mesh = example_mesh.rotate(r.as_matrix())
        example_mesh.compute_triangle_normals()
        # plot_meshes(example_mesh)

        ga, colored_icosahedron = visualize_gaussian_integration(ga_py, example_mesh)
        # plot_projection(ga)
        # plot_hilbert_curve(ga)
        plot_meshes(colored_icosahedron, example_mesh)


if __name__ == "__main__":
    main()
