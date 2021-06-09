import numpy as np
import scipy.ndimage as ndi
from .rank_order import rank_order

def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities)
    coord = np.transpose(coord)[idx_maxsort]
    # select num_peaks peaks
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]
    return coord


def _get_peak_mask(image, min_distance, footprint, threshold_abs,
                   threshold_rel):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint is not None:
        image_max = ndi.maximum_filter(image, footprint=footprint,
                                       mode='constant')
    else:
        size = 2 * min_distance + 1
        image_max = ndi.maximum_filter(image, size=size, mode='constant')
    mask = image == image_max
    if threshold_rel is not None:
        threshold = max(threshold_abs, threshold_rel * image.max())
    else:
        threshold = threshold_abs
    mask &= image > threshold
    return mask


def _exclude_border(mask, exclude_border):
    """
    Remove peaks near the borders
    """
    # zero out the image borders
    for i, excluded in enumerate(exclude_border):
        if excluded == 0:
            continue
        mask[(slice(None),) * i + (slice(None, excluded),)] = False
        mask[(slice(None),) * i + (slice(-excluded, None),)] = False
    return mask


def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf):
    """Find peaks in an image as coordinate list or boolean mask.
    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).
    If there are multiple local maxima with identical pixel intensities
    inside the region defined with `min_distance`,
    the coordinates of all such pixels are returned.
    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.
    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates. The coordinates are sorted according to peaks
        values (Larger first). If False, the output will be a boolean
        array shaped as `image.shape` with peaks present at True
        elements.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.
    Returns
    -------
    output : ndarray or ndarray of bools
        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.
    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.
    See also
    --------
    skimage.feature.corner_peaks
    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])
    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])
    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_local_max(img2, exclude_border=0)
    array([[10, 10, 10]])
    """
    out = np.zeros_like(image, dtype=bool)

    threshold_abs = threshold_abs if threshold_abs is not None else image.min()

    if isinstance(exclude_border, bool):
        exclude_border = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        exclude_border = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError(
                    "`exclude_border` cannot contain a negative value")
    else:
        raise TypeError(
            "`exclude_border` must be bool, int, or tuple with the same "
            "length as the dimensionality of the image.")

    # no peak for a trivial image
    if np.all(image == image.flat[0]):
        if indices is True:
            return np.empty((0, image.ndim), np.int)
        else:
            return out

    # In the case of labels, call ndi on each label
    if labels is not None:
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        # create a mask for the non-exclude region
        inner_mask = _exclude_border(np.ones_like(labels, dtype=bool),
                                     exclude_border)

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks and mark them in
        # variable out.
        for label_idx, obj in enumerate(ndi.find_objects(labels)):
            img_object = image[obj] * (labels[obj] == label_idx + 1)
            mask = _get_peak_mask(img_object, min_distance, footprint,
                                  threshold_abs, threshold_rel)
            if exclude_border:
                # remove peaks fall in the exclude region
                mask &= inner_mask[obj]
            coordinates = _get_high_intensity_peaks(img_object, mask,
                                                    num_peaks_per_label)
            nd_indices = tuple(coordinates.T)
            mask.fill(False)
            mask[nd_indices] = True
            out[obj] += mask

        if not indices and np.isinf(num_peaks):
            return out

        coordinates = _get_high_intensity_peaks(image, out, num_peaks)
        if indices:
            return coordinates
        else:
            out.fill(False)
            nd_indices = tuple(coordinates.T)
            out[nd_indices] = True
            return out

    # Non maximum filter
    mask = _get_peak_mask(image, min_distance, footprint, threshold_abs,
                          threshold_rel)

    mask = _exclude_border(mask, exclude_border)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(image, mask, num_peaks)

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out
