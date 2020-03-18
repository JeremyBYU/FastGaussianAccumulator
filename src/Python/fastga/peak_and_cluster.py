import time
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from skimage.feature import peak_local_max

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted,
               find_peaks_kwargs=dict(height=0.05, threshold=None, distance=4, width=None, prominence=0.07),
               cluster_kwargs=dict(t=0.15, criterion='distance')):
    peaks, _ = find_peaks(accumulator_normalized_sorted, **find_peaks_kwargs)
    gaussian_normal_1d_clusters = gaussian_normals_sorted[peaks,:]
    Z = linkage(gaussian_normal_1d_clusters, 'single')
    clusters = fcluster(Z, **cluster_kwargs)
    weights_1d_clusters = accumulator_normalized_sorted[peaks]
    average_peaks, average_weights = average_clusters(gaussian_normal_1d_clusters, weights_1d_clusters, clusters)
    return peaks, clusters, average_peaks, average_weights

def get_high_intensity_peaks(image, mask, num_peaks=np.inf):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    coord = np.column_stack(coord)
    # Highest peak first
    return coord[::-1]

def find_peaks_from_ico_charts(ico_charts, normalized_bucket_counts_by_vertex,
               find_peaks_kwargs=dict(threshold_abs=15, min_distance=1, exclude_border=False, indices=False),
               cluster_kwargs=dict(t=0.05, criterion='distance')):
    # Get data from ico chart
    # t0 = time.perf_counter()
    image_to_vertex_idx = np.asarray(ico_charts.image_to_vertex_idx)
    image = np.asarray(ico_charts.image)
    mask = np.asarray(ico_charts.mask)
    mask = np.ma.make_mask(mask, copy=False)
    vertices_mesh = np.asarray(ico_charts.sphere_mesh.vertices)
    # 2D Peak Detection
    peak_mask = peak_local_max(image, **find_peaks_kwargs)
    # Filter out invalid peaks, get associated normals
    valid_peaks = mask & peak_mask
    peak_image_idx = get_high_intensity_peaks(image, valid_peaks)
    vertices_idx = image_to_vertex_idx[peak_image_idx[:, 0], peak_image_idx[:, 1]]
    unclustered_peak_normals = vertices_mesh[vertices_idx,:]
    # t1 = time.perf_counter()
    # print(peak_image_idx)

    Z = linkage(unclustered_peak_normals, 'single')
    clusters = fcluster(Z, **cluster_kwargs)
    # t2 = time.perf_counter()

    weights_1d_clusters = normalized_bucket_counts_by_vertex[vertices_idx]
    average_peaks, average_weights = average_clusters(unclustered_peak_normals, weights_1d_clusters, clusters, average_filter=dict(min_total_weight=0.10))

    # print("IcoChart Peak Detection - Find Peaks Execution Time (ms): {:.1f}; Hierarchical Clustering Execution Time (ms): {:.1f}".format((t1-t0) * 1000, (t2-t1) * 1000))
    return vertices_idx, clusters, average_peaks, average_weights

def get_point_clusters(points, point_weights, clusters):
    point_clusters = []
    cluster_groups = np.unique(clusters)
    for cluster in cluster_groups:
        temp_mask = clusters == cluster
        point_clusters.append((points[temp_mask, :], point_weights[temp_mask]))
    return point_clusters

def average_clusters(peaks, peak_weights, clusters, average_filter=dict(min_total_weight=0.2)):
    cluster_points = get_point_clusters(peaks, peak_weights, clusters)
    clusters_averaged = []
    clusters_total_weight = []
    for points, point_weights in cluster_points:
        total_weight = np.sum(point_weights)
        avg_point = np.average(points, axis=0, weights=point_weights)
        if total_weight < average_filter['min_total_weight']:
            continue
        clusters_averaged.append(avg_point)
        clusters_total_weight.append(total_weight)
    normals = np.array(clusters_averaged)
    weights = np.array(clusters_total_weight)
    normals, _ = normalized(normals)
    idx = np.argsort(weights)[::-1]
    normals = normals[idx]
    weights = weights[idx]

    return normals, weights