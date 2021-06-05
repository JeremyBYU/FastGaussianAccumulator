import numpy as np
import pytest
from tests.python.helpers.setup_helper import setup_fastga, setup_fastga_simple, cluster_normals, sort_by_distance_from_point
from fastga.scikit_image.skimage_feature_peak import peak_local_max
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from fastga import MatX3d
from scipy.spatial.distance import cdist

np.random.seed(1)

@pytest.mark.parametrize("num_clusters", range(1, 10))
@pytest.mark.parametrize("normals_per_cluster", [100, 1000, 10000])
def test_same_output(num_clusters, normals_per_cluster):
    fixture = setup_fastga_simple(level=3)
    clusters, normals = cluster_normals(num_clusters=num_clusters, normals_per_cluster=normals_per_cluster, patch_deg=5)
    clusters =np.concatenate(clusters)
    ga = fixture['ga']
    ico = fixture['ico']
    ga.integrate(MatX3d(clusters))

    # New API
    new_api_peaks = np.array(ga.find_peaks(threshold_abs=20, cluster_distance=0.1, min_cluster_weight=0.0))

    # Sort the peaks
    new_api_peaks = sort_by_distance_from_point(new_api_peaks)
    gt_peaks = sort_by_distance_from_point(normals)

    print(new_api_peaks)
    print(gt_peaks)
    try:
        np.testing.assert_allclose(new_api_peaks, gt_peaks, atol=0.2)
    except:
        if new_api_peaks.shape == gt_peaks.shape:
            dist = cdist(new_api_peaks, gt_peaks)
            idx = np.argmin(dist, axis=0)
            gt_peaks = gt_peaks[idx]
            np.testing.assert_allclose(new_api_peaks, gt_peaks, atol=0.2)
        else:
            raise
