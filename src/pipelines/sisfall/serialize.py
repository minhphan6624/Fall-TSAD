import numpy as np
from pathlib import Path
from .segment_data import segment_dataset

def serialize(normed_splits: dict, out_dir: Path, window: int, overlap: int, sampling_rate: int):
    """
    Segments the normalized data and saves the results to disk.

    Args:
        normed_splits (dict): A dictionary with keys 'train', 'val', 'test', where each
                              value is a list of tuples (normalized_data, metadata_row).
        out_dir (Path): The directory to save the processed data to.
        window (int): The size of the segmentation window.
        overlap (int): The overlap between consecutive windows.
        sampling_rate (int): The sampling rate of the data in Hz.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data_and_metadata in normed_splits.items():
        # Unzip the list of tuples into two lists: one for data, one for metadata rows
        data_list, metadata_rows = zip(*data_and_metadata)
        
        # Segment the data and apply impact-zone-based labeling
        segmented_data, segmented_labels = segment_dataset(
            data_list=list(data_list),
            metadata_rows=list(metadata_rows),
            window_size=window,
            overlap=overlap,
            sampling_rate=sampling_rate
        )
        
        # Save the segmented data and labels
        np.save(out_dir / f"{split}_data.npy", segmented_data)
        np.save(out_dir / f"{split}_labels.npy", segmented_labels)
