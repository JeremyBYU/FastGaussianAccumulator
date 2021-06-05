import numpy as np
import pytest
from tests.python.helpers.setup_helper import setup_fastga, setup_fastga_simple, cluster_normals, sort_by_distance_from_point
from fastga.scikit_image.skimage_feature_peak import peak_local_max
from fastga.peak_and_cluster import find_peaks_from_ico_charts
from fastga import MatX3d

np.random.seed(1)

def find_peaks_from_ico_charts_updated(ga, ico, find_peaks_kwargs, cluster_kwargs, average_filter):
    normalized_bucket_counts_by_vertex = np.asarray(ga.get_normalized_bucket_counts_by_vertex(True))
    average_vertex_normals = np.asarray(ga.get_average_normals_by_vertex(True))
    res = find_peaks_from_ico_charts(ico, normalized_bucket_counts_by_vertex, 
                                    vertices=average_vertex_normals, find_peaks_kwargs=find_peaks_kwargs,
                                    cluster_kwargs=cluster_kwargs, average_filter=average_filter)
    return res


@pytest.mark.parametrize("level", [2, 3, 4])
def test_peak_local_max_scipy(benchmark, small_normals, level, find_peaks_kwargs):
    fixture = setup_fastga(small_normals, level=level)
    ico = fixture['ico']

    image = np.asarray(ico.image)
    # 2D Peak Detection
    peak_mask = benchmark(peak_local_max, image, **find_peaks_kwargs)

@pytest.mark.parametrize("level", [2, 3, 4])
def test_peak_local_max_cpp(benchmark, small_normals, level, find_peaks_kwargs):
    fixture = setup_fastga(small_normals, level=level)
    ico = fixture['ico']

    find_peaks_kwargs_2 = { your_key: find_peaks_kwargs[your_key] for your_key in ['threshold_abs', 'exclude_border'] }
    # 2D Peak Detection
    peak_mask = benchmark(ico.find_peaks, **find_peaks_kwargs_2)

@pytest.mark.parametrize("level", [2, 3, 4])
def test_peak_all_scipy(benchmark, small_normals, level, find_peaks_kwargs, cluster_kwargs, average_filter):
    fixture = setup_fastga(small_normals, level=level)
    ga = fixture['ga']
    ico = fixture['ico']

    image = np.asarray(ico.image)
    # 2D Peak Detection
    peak_mask = benchmark(find_peaks_from_ico_charts_updated, ga, ico, find_peaks_kwargs, cluster_kwargs, average_filter )

@pytest.mark.parametrize("level", [2, 3, 4])
def test_peak_all_cpp(benchmark, small_normals, level, find_peaks_kwargs, cluster_kwargs, average_filter):
    fixture = setup_fastga(small_normals, level=level)
    ga = fixture['ga']
    ico = fixture['ico']

    find_peaks_kwargs_2 = { your_key: find_peaks_kwargs[your_key] for your_key in ['threshold_abs', 'exclude_border'] }
    # 2D Peak Detection
    peak_mask = benchmark(ga.find_peaks,**find_peaks_kwargs_2)

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
    new_api_peaks = np.array(ga.find_peaks(threshold_abs=20, cluster_distance=0.1, min_cluster_weight=0.1))

    # Old API
    normalized_bucket_counts_by_vertex = ga.get_normalized_bucket_counts_by_vertex(True)
    average_vertex_normals = np.asarray(ga.get_average_normals_by_vertex(True))
    ico.fill_image(normalized_bucket_counts_by_vertex)
    _, _, old_api_peaks, _ = find_peaks_from_ico_charts(ico, np.asarray(normalized_bucket_counts_by_vertex), vertices=average_vertex_normals)

    # Sort the peaks
    new_api_peaks = sort_by_distance_from_point(new_api_peaks)
    old_api_peaks = sort_by_distance_from_point(old_api_peaks)

    print(new_api_peaks)
    print(old_api_peaks)

    np.testing.assert_allclose(new_api_peaks, old_api_peaks)

