import numpy as np

def segment_data(data: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
    """
    Segments time-series data into overlapping windows.

    Args:
        data (np.ndarray): The input time-series data (e.g., (num_samples, num_features)).
        window_size (int): The size of each segment window.
        overlap (int): The number of samples that overlap between consecutive windows.

    Returns:
        np.ndarray: A 3D array of segmented windows (num_windows, window_size, num_features).
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("Overlap must be non-negative and less than window_size.")

    step_size = window_size - overlap
    num_samples, num_features = data.shape

    # Calculate the number of windows. Ensure that we don't create partial windows at the end
    
    num_windows = (num_samples - window_size) // step_size + 1
    if num_windows <= 0: # Not enough samples to create even one full window
        return np.empty((0, window_size, num_features), dtype=data.dtype)

    # Initialize the array to hold the segments
    segments = np.zeros((num_windows, window_size, num_features), dtype=data.dtype)

    # Extract segments
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        segments[i] = data[start_idx:end_idx]

    return segments

def segment_dataset(
    data_list: list[np.ndarray], 
    labels_list: list[int], 
    window_size: int, 
    overlap: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments a list of data arrays and their corresponding labels into windows.

    Args:
        data_list (list[np.ndarray]): A list of individual time-series data arrays.
        labels_list (list[int]): A list of labels corresponding to each data array.
        window_size (int): The size of each segment window.
        overlap (int): The number of samples that overlap between consecutive windows.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - segmented_data (np.ndarray): All segmented windows concatenated.
            - segmented_labels (np.ndarray): Labels for each segmented window.
    """
    all_segments = []
    all_labels = []

    for i, data in enumerate(data_list):
        segments = segment_data(data, window_size, overlap)
        if segments.shape[0] > 0:
            all_segments.append(segments)
            # Each segment from this data array gets the same label
            all_labels.append(np.full(segments.shape[0], labels_list[i]))

    if not all_segments:
        return np.empty((0, window_size, data_list[0].shape[1])), np.empty(0)

    return np.concatenate(all_segments, axis=0), np.concatenate(all_labels, axis=0)
