import numpy as np

def filter_and_normalize( data, lower_percentile=1, upper_percentile=99):
    """
    Filters outliers and normalizes the data.
        data (np.array): Input data to be filtered and normalized.
        lower_percentile (float): Lower percentile for filtering.
        upper_percentile (float): Upper percentile for filtering.
    Return:
        np.array: Filtered and normalized data.
    """
    # Compute the lower and upper bounds for filtering
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)

    # Clip the data to remove outliers
    data_clipped = np.clip(data, lower_bound, upper_bound)

    # Normalize the data to range [0, 1]
    data_normalized = (data_clipped - data_clipped.min()) / (data_clipped.max() - data_clipped.min())

    return data_normalized


def crop_3d_spatial_bounding_box( img, threshold=0.02):
    """
    crop a 3d + t numpy array
    :param img: 3d+t numpy array
    :param threshold: values below this threshold do not count
    :return: cropped image
    """
    # Validate input dimensions
    assert len(img.shape) == 4, "Input image must have 4 dimensions [W, H, D, T]"

    # Find the bounding box that includes all regions with values greater than the threshold
    mask = np.max(img, axis=3) > threshold  # Aggregate across the time axis

    # Get the slices for width, height, depth where the value exceeds the threshold
    w_indices = np.any(mask, axis=(1, 2))  # Collapse H and D to find relevant W
    h_indices = np.any(mask, axis=(0, 2))  # Collapse W and D to find relevant H
    d_indices = np.any(mask, axis=(0, 1))  # Collapse W and H to find relevant D

    # Find the bounds for width, height, and depth
    min_w, max_w = np.where(w_indices)[0][[0, -1]]
    min_h, max_h = np.where(h_indices)[0][[0, -1]]
    min_d, max_d = np.where(d_indices)[0][[0, -1]]

    # Crop the image across width, height, and depth, keeping time constant
    cropped_img = img[min_w:max_w + 1, min_h:max_h + 1, min_d:max_d + 1, :]

    return cropped_img