import numpy as np

def down_sample_normals(triangle_normals, down_sample_fraction=0.12, min_samples=10000, flip_normals=False, **kwargs):
    """Return dowsampled normals

    Args:
        triangle_normals (np.ndarray): Triangle Normals
        down_sample_fraction (float, optional): Fraction to downsample. Defaults to 0.12.
        min_samples (int, optional): Minimum number of samples. Defaults to 10000.
        flip_normals (bool, optional): Reverse the normals?. Defaults to False.

    Returns:
        np.ndarray: NX3 downsampled normals
    """
    num_normals = triangle_normals.shape[0]
    to_sample = int(down_sample_fraction * num_normals)
    to_sample = max(min([num_normals, min_samples]), to_sample)
    ds_step = int(num_normals / to_sample)
    triangle_normals_ds = np.ascontiguousarray(triangle_normals[:num_normals:ds_step, :])
    if flip_normals:
        triangle_normals_ds = triangle_normals_ds * -1.0
    return triangle_normals_ds