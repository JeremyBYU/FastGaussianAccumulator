from pathlib import Path
import pytest
import numpy as np


np.random.seed(1)

@pytest.fixture
def small_normals():
    normals = np.asarray([
        [0.0, 0.0, 0.95],
        [0.0, 0.0, 0.98],
        [0.95, 0.0, 0],
        [0.98, 0.0, 0],
    ])
    return normals

@pytest.fixture
def find_peaks_kwargs():
    find_peaks_kwargs = dict(threshold_abs=20, min_distance=1, exclude_border=False, indices=False)
    return find_peaks_kwargs

@pytest.fixture
def cluster_kwargs():
    return dict(t=0.10, criterion='distance')

@pytest.fixture
def average_filter():
   return dict(min_total_weight=0.15)

                            

