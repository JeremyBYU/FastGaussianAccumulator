import numpy as np
import time 
import matplotlib.pyplot as plt
import open3d as o3d
from fastga import GaussianAccumulatorS2, MatX3d, refine_icosahedron, refine_icochart
from src.Python.slowga import (GaussianAccumulatorKDPy, filter_normals_by_phi, get_colors,
                               create_open_3d_mesh, assign_vertex_colors, plot_meshes, find_peaks_from_accumulator)


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

    vertex = vertices[0,:]
    return chart_mesh, chart_start_idx, chart_end_idx

def decompose(ico):
    triangles = np.asarray(ico.triangles)
    vertices = np.asarray(ico.vertices)
    ico_o3d = create_open_3d_mesh(triangles, vertices)
    return triangles, vertices, ico_o3d

def main():
    ico = refine_icosahedron(level=0)
    ico_s2 = GaussianAccumulatorS2(level=1)
    ico_s2_organized_mesh = ico_s2.copy_ico_mesh(True)
    triangles_ico, vertices, ico_o3d = decompose(ico)
    triangles_s2_om, _, ico_o3d_s2_om = decompose(ico_s2_organized_mesh)
    icochart = refine_icochart(level=1)
    _, _, flatchart_o3d = decompose(icochart)

    colors_vertices = o3d.utility.Vector3dVector(get_colors(range(vertices.shape[0]), colormap=plt.cm.Paired)[:, :3])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = colors_vertices


    colors = get_colors(range(triangles_ico.shape[0]), colormap=plt.cm.tab20)[:, :3]
    colors_s2 = get_colors(range(triangles_s2_om.shape[0]), colormap=plt.cm.tab20)[:, :3]
    # To verify colormapping
    colors_s2 = np.vstack((colors_s2[::4,:], colors_s2[::4,:], colors_s2[::4,:], colors_s2[::4,:]))


    colored_ico = assign_vertex_colors(ico_o3d, colors)
    colored_ico_s2 = assign_vertex_colors(ico_o3d_s2_om, colors_s2)
    colored_icochart, start_idx, end_idx = extract_chart(colored_ico_s2, chart_idx=4)
    colored_flatchart = assign_vertex_colors(flatchart_o3d, colors_s2[start_idx:end_idx, :])

    # plot_meshes([colored_ico, pcd])
    plot_meshes([colored_ico, pcd], [colored_ico_s2, pcd], colored_icochart, colored_flatchart)
    # o3d.visualization.draw_geometries([ico_o3d])

    

if __name__ == "__main__":
    main()