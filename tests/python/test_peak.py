import numpy as np
from tests.python.helpers.setup_helper import setup_fastga
from fastga.scikit_image.skimage_feature_peak import peak_local_max
from fastga.peak_and_cluster import find_peaks_from_ico_charts


def find_peaks_from_ico_charts_updated(ga, ico, find_peaks_kwargs, cluster_kwargs, average_filter):
    normalized_bucket_counts_by_vertex = np.asarray(ga.get_normalized_bucket_counts_by_vertex(True))
    average_vertex_normals = np.asarray(ga.get_average_normals_by_vertex(True))
    res = find_peaks_from_ico_charts(ico, normalized_bucket_counts_by_vertex, 
                                    vertices=average_vertex_normals, find_peaks_kwargs=find_peaks_kwargs,
                                    cluster_kwargs=cluster_kwargs, average_filter=average_filter)
    return res



def test_peak_local_max_scipy(benchmark, small_normals, find_peaks_kwargs):
    fixture = setup_fastga(small_normals, level=4)
    ico = fixture['ico']

    image = np.asarray(ico.image)
    # 2D Peak Detection
    peak_mask = benchmark(peak_local_max, image, **find_peaks_kwargs)

def test_peak_local_max_cpp(benchmark, small_normals, find_peaks_kwargs):
    fixture = setup_fastga(small_normals, level=4)
    ico = fixture['ico']

    find_peaks_kwargs_2 = { your_key: find_peaks_kwargs[your_key] for your_key in ['threshold_abs', 'exclude_border'] }
    # 2D Peak Detection
    peak_mask = benchmark(ico.find_peaks, **find_peaks_kwargs_2)


def test_peak_all_scipy(benchmark, small_normals, find_peaks_kwargs, cluster_kwargs, average_filter):
    fixture = setup_fastga(small_normals, level=4)
    ga = fixture['ga']
    ico = fixture['ico']

    image = np.asarray(ico.image)
    # 2D Peak Detection
    peak_mask = benchmark(find_peaks_from_ico_charts_updated, ga, ico, find_peaks_kwargs, cluster_kwargs, average_filter )

def test_peak_all_cpp(benchmark, small_normals, find_peaks_kwargs, cluster_kwargs, average_filter):
    fixture = setup_fastga(small_normals, level=4)
    ga = fixture['ga']
    ico = fixture['ico']

    find_peaks_kwargs_2 = { your_key: find_peaks_kwargs[your_key] for your_key in ['threshold_abs', 'exclude_border'] }
    # 2D Peak Detection
    peak_mask = benchmark(ga.find_peaks_from_ico_charts, ico, **find_peaks_kwargs_2)