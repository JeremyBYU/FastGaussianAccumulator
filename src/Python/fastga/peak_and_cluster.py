
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster

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