import numpy as np

from fastga import GaussianAccumulatorS2Beta, IcoCharts, MatX3d

from scipy.spatial.transform import Rotation as R

    
def setup_fastga(normals:np.ndarray, level=4):
    kwargs_s2 = dict(level=level)
    # Create Gaussian Accumulator
    ga_cpp_s2 = GaussianAccumulatorS2Beta(**kwargs_s2)
    _ = ga_cpp_s2.integrate(MatX3d(normals))
  
    ico_chart_ = IcoCharts(level)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    return dict(ga=ga_cpp_s2, ico=ico_chart_, normals=normals)


def polar_to_catersian(theta, phi):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    normals = np.column_stack([x, y, z])
    return normals


def cartesian_to_polar(x, y, z):
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return np.column_stack([theta, phi])


def sample_sphere_cap(n=100, deg=10, normal=[1, 0, 0]):
    min_z = np.cos(np.radians(deg))
    z = np.random.uniform(min_z, 1.0, n)
    r = np.sqrt(1 - z * z)
    theta = np.random.uniform(0, 1.0, n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    rm, _, = R.align_vectors(np.array([normal]), np.array([[0.0, 0.0, 1.0]]))
    points = np.column_stack([x, y, z])
    points = rm.apply(points)

    return points

def generate_random_normals(n=1, umin=0.0, umax=1.0, vmin=0.0, vmax=1.0):
    normals = np.zeros((n, 3))
    uniform_number_1 = np.random.uniform(umin, umax, n)
    uniform_number_2 = np.random.uniform(vmin, vmax, n)
    theta = 2 * np.pi * uniform_number_1
    phi = np.arccos(2 * uniform_number_2 - 1)
    normals = polar_to_catersian(theta, phi)
    polars = np.column_stack([theta, phi])
    return normals, polars


def cluster_normals(num_clusters=2, normals_per_cluster=5, patch_deg=10, normals=None):
    if normals is not None:
        num_clusters = normals.shape[0]
    else:
        normals, _ = generate_random_normals(num_clusters)
  
    clusters = []
    for i in range(num_clusters):
        normal = normals[i]
        cluster_normals = sample_sphere_cap(n=normals_per_cluster, deg=patch_deg, normal=normal.tolist())
        clusters.append(cluster_normals)
    return clusters, normals

def sort_by_distance_from_point(array, point=[0.0, 0.0, 1.0]):
    diff = array - point
    diff = np.sum(np.power(diff, 2), axis=1)
    idx = np.argsort(diff)[::-1]
    return array[idx]
