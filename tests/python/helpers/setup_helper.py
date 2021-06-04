import numpy as np

from fastga import GaussianAccumulatorS2Beta, IcoCharts, MatX3d

    
def setup_fastga(normals:np.ndarray, level=4):
    kwargs_s2 = dict(level=level)
    # Create Gaussian Accumulator
    ga_cpp_s2 = GaussianAccumulatorS2Beta(**kwargs_s2)
    _ = ga_cpp_s2.integrate(MatX3d(normals))
  
    ico_chart_ = IcoCharts(level)
    normalized_bucket_counts_by_vertex = ga_cpp_s2.get_normalized_bucket_counts_by_vertex(True)
    ico_chart_.fill_image(normalized_bucket_counts_by_vertex)

    return dict(ga=ga_cpp_s2, ico=ico_chart_, normals=normals)

    # # 2D Peak Detection
    # find_peaks_kwargs = dict(threshold_abs=20, min_distance=1, exclude_border=False, indices=False)
    # cluster_kwargs = dict(t=0.2, criterion='distance')
    # average_filter = dict(min_total_weight=0.05)
    # _, _, avg_peaks, _ = find_peaks_from_ico_charts(ico_chart_, np.asarray(normalized_bucket_counts_by_vertex), find_peaks_kwargs=find_peaks_kwargs, cluster_kwargs=cluster_kwargs)
    # print("Detected Peaks: {}".format(avg_peaks))