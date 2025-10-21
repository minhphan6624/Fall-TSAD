import numpy as np
import pandas as pd

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
    metadata_rows: list[pd.Series], # Changed from labels_list to metadata_rows
    window_size: int, 
    overlap: int,
    sampling_rate: int # New parameter
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments a list of data arrays and their corresponding labels into windows,
    applying impact-zone-based labeling for fall events.

    Args:
        data_list (list[np.ndarray]): A list of individual time-series data arrays.
        metadata_rows (list[pd.Series]): A list of metadata rows (Series) corresponding to each data array.
        window_size (int): The size of each segment window.
        overlap (int): The number of samples that overlap between consecutive windows.
        sampling_rate (int): The sampling rate of the data in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - segmented_data (np.ndarray): All segmented windows concatenated.
            - segmented_labels (np.ndarray): Labels for each segmented window.
    """
    all_segments = []
    all_labels = []
    
    impact_pre_samples = int(0.5 * sampling_rate) # 0.5 seconds before impact
    impact_post_samples = int(1.0 * sampling_rate) # 1.0 seconds after impact

    for i, data in enumerate(data_list):
        metadata = metadata_rows[i]
        segments = segment_data(data, window_size, overlap)
        
        if segments.shape[0] > 0:
            all_segments.append(segments)
            
            if metadata["is_fall"] == 1:
                # Fall event: apply impact-zone-based labeling
                magnitude = np.sqrt(np.sum(data**2, axis=1))
                impact_idx = np.argmax(magnitude) # Peak magnitude is impact time t*

                window_labels = np.zeros(segments.shape[0], dtype=int)
                step_size = window_size - overlap

                for j in range(segments.shape[0]):
                    window_start_idx = j * step_size
                    window_end_idx = window_start_idx + window_size

                    # Check for overlap with impact zone [t* - 0.5s, t* + 1.0s]
                    if (window_end_idx > (impact_idx - impact_pre_samples)) and \
                       (window_start_idx < (impact_idx + impact_post_samples)):
                        window_labels[j] = 1
                all_labels.append(window_labels)
            else:
                # ADL event: all windows are labeled 0
                all_labels.append(np.full(segments.shape[0], 0, dtype=int))

    if not all_segments:
        # Handle case where data_list might be empty or segments.shape[0] is always 0
        # Ensure data_list[0].shape[1] is safely accessed
        num_features = data_list[0].shape[1] if data_list else 0
        return np.empty((0, window_size, num_features)), np.empty(0)

    return np.concatenate(all_segments, axis=0), np.concatenate(all_labels, axis=0)
