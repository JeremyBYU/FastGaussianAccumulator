import numpy as np
import open3d as o3d

GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0
ICOSAHEDRON_TRUE_RADIUS = np.sqrt(1 + np.power(GOLDEN_RATIO, 2))
ICOSAHEDRON_SCALING = 1.0 / ICOSAHEDRON_TRUE_RADIUS


def create_icosahedron(radius=ICOSAHEDRON_SCALING):
    if (radius <= 0):
        ValueError("Radius too small")

    p = GOLDEN_RATIO
    vertices = []
    triangles = []
    vertices.append(radius * np.array([-1, 0, p]))
    vertices.append(radius * np.array([1, 0, p]))
    vertices.append(radius * np.array([1, 0, -p]))
    vertices.append(radius * np.array([-1, 0, -p]))
    vertices.append(radius * np.array([0, -p, 1]))
    vertices.append(radius * np.array([0, p, 1]))
    vertices.append(radius * np.array([0, p, -1]))
    vertices.append(radius * np.array([0, -p, -1]))
    vertices.append(radius * np.array([-p, -1, 0]))
    vertices.append(radius * np.array([p, -1, 0]))
    vertices.append(radius * np.array([p, 1, 0]))
    vertices.append(radius * np.array([-p, 1, 0]))
    triangles.append(np.array([0, 4, 1]))
    triangles.append(np.array([0, 1, 5]))
    triangles.append(np.array([1, 4, 9]))
    triangles.append(np.array([1, 9, 10]))
    triangles.append(np.array([1, 10, 5]))
    triangles.append(np.array([0, 8, 4]))
    triangles.append(np.array([0, 11, 8]))
    triangles.append(np.array([0, 5, 11]))
    triangles.append(np.array([5, 6, 11]))
    triangles.append(np.array([5, 10, 6]))
    triangles.append(np.array([4, 8, 7]))
    triangles.append(np.array([4, 7, 9]))
    triangles.append(np.array([3, 6, 2]))
    triangles.append(np.array([3, 2, 7]))
    triangles.append(np.array([2, 6, 10]))
    triangles.append(np.array([2, 10, 9]))
    triangles.append(np.array([2, 9, 7]))
    triangles.append(np.array([3, 11, 6]))
    triangles.append(np.array([3, 8, 11]))
    triangles.append(np.array([3, 7, 8]))
    triangles = np.array(triangles)
    vertices = np.array(vertices)
    return triangles, vertices


def create_icosahedron_mesh(radius=ICOSAHEDRON_SCALING):
    triangles, vertices = create_icosahedron(radius=radius)
    # mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=ICOSAHEDRON_SCALING)
    return mesh


def cantor_mapping(k1, k2):
    return int(((k1 + k2) * (k1 + k2 + 1)) / 2.0 + k2)


def generate_key_from_point(p1_idx, p2_idx):
    lower_idx, higher_idx = (
        p1_idx, p2_idx) if p1_idx < p2_idx else (p2_idx, p1_idx)
    return cantor_mapping(lower_idx, higher_idx)


def construct_midpoint(p1_idx, p2_idx, vertices):
    p1 = vertices[p1_idx]
    p2 = vertices[p2_idx]
    midpoint_on_plane = (p2 + p1) / 2.0
    scaling = 1 / np.linalg.norm(midpoint_on_plane)
    midpoint_on_sphere = midpoint_on_plane * scaling
    return midpoint_on_sphere


def get_point_idx(p1_idx, p2_idx, point_to_idx_map, vertices):
    point_key = generate_key_from_point(p1_idx, p2_idx)
    if point_to_idx_map.get(point_key):
        # Existing point Idx
        return point_to_idx_map[point_key]
    else:
        # New point idx
        point_to_idx_map[point_key] = len(vertices)
        midpoint_on_sphere = construct_midpoint(p1_idx, p2_idx, vertices)
        vertices.append(midpoint_on_sphere)
        return point_to_idx_map[point_key]


def refine_icosahedron(triangles, vertices, level=2):
    vertices = list(vertices)
    triangles = triangles.tolist()
    # HashMap that maps a midpoint (identified by two point indexes) to its own point index
    point_to_idx_map = dict()
    for i in range(level):
        triangles_refined = []
        # loop through every triangle and create 4 new ones based upon midpoints
        for triangle in triangles:
            p1_idx = triangle[0]
            p2_idx = triangle[1]
            p3_idx = triangle[2]

            # Create new points (if not existing) and return point index
            p4_idx = get_point_idx(p1_idx, p2_idx, point_to_idx_map, vertices)
            p5_idx = get_point_idx(p2_idx, p3_idx, point_to_idx_map, vertices)
            p6_idx = get_point_idx(p3_idx, p1_idx, point_to_idx_map, vertices)
            # Create the four new triangles
            t1 = [p1_idx, p4_idx, p6_idx]
            t2 = [p4_idx, p2_idx, p5_idx]
            t3 = [p6_idx, p5_idx, p3_idx]
            t4 = [p6_idx, p4_idx, p5_idx]
            # Append triangles to the new refined triangle array
            triangles_refined.extend([t1, t2, t3, t4])
        # overwrite existing triangles with this new array
        triangles = triangles_refined
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    return vertices, triangles
