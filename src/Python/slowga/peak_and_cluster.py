import time
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from .helper import normalized


def find_peaks_from_accumulator(gaussian_normals_sorted, accumulator_normalized_sorted,
               find_peaks_kwargs=dict(height=0.05, threshold=None, distance=4, width=None, prominence=0.07),
               cluster_kwargs=dict(t=0.15, criterion='distance')):
    t0 = time.perf_counter()
    peaks, _ = find_peaks(accumulator_normalized_sorted, **find_peaks_kwargs)
    t1 = time.perf_counter()

    gaussian_normal_1d_clusters = gaussian_normals_sorted[peaks,:]
    Z = linkage(gaussian_normal_1d_clusters, 'single')
    clusters = fcluster(Z, **cluster_kwargs)
    t2 = time.perf_counter()

    weights_1d_clusters = accumulator_normalized_sorted[peaks]
    average_peaks, average_weights = average_clusters(gaussian_normal_1d_clusters, weights_1d_clusters, clusters)

    print("Peak Detection - Find Peaks Execution Time (ms): {:.1f}; Hierarchical Clustering Execution Time (ms): {:.1f}".format((t1-t0) * 1000, (t2-t1) * 1000))
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
    normals, _ = normalized(normals)

    return normals, np.array(clusters_total_weight)

