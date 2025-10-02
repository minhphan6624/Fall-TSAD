from pathlib import Path
from .paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from .build_metadata import build_metadata
from .loader import load_sisfall_file
from .split import split_data_by_subject
from .segment import segment_dataset
from .normalize import StandardScaler
from .serialize import save_processed_data

import numpy as np

# -- Configuration --
WINDOW_SIZE = 600
OVERLAP = 300
RANDOM_STATE = 42

def main():
    # -- 1. Extract and split meta data --
    metadata_df = build_metadata(RAW_DATA_DIR)
    train_meta, val_meta, test_meta = split_data_by_subject(metadata_df, 
                                                            random_state=RANDOM_STATE)
    
    train_meta = train_meta[train_meta['is_fall'] == 0].reset_index(drop=True)

    # -- 2. Load data for each split based on metadata --
    def load_data_for_split(meta_df):
        data_list = []
        for _, row in meta_df.iterrows():
            data = load_sisfall_file(Path(row['path'])) # Load data file based on the path
            data_list.append(data)

        return data_list

    train_raw_data = load_data_for_split(train_meta)
    val_raw_data = load_data_for_split(val_meta)
    test_raw_data = load_data_for_split(test_meta)

    # -- 3. Normalization --

    # Concatenate all training data into a numpy ndarray to fit the scaler
    full_train_data = np.concatenate(train_raw_data, axis=0)

    # Initialize and fit the scaler on training data only
    scaler = StandardScaler()
    scaler.fit(full_train_data)

    # Apply normalization to each split
    train_norm_data = [scaler.transform(data) for data in train_raw_data]
    val_norm_data = [scaler.transform(data) for data in val_raw_data]
    test_norm_data = [scaler.transform(data) for data in test_raw_data]

    # -- 4. Segmentation --
    train_segments = segment_dataset(
        train_norm_data, train_meta['is_fall'].tolist(), WINDOW_SIZE, OVERLAP)
    val_segments = segment_dataset(
        val_norm_data, val_meta['is_fall'].tolist(), WINDOW_SIZE, OVERLAP)
    test_segments = segment_dataset(
        test_norm_data, test_meta['is_fall'].tolist(), WINDOW_SIZE, OVERLAP)

    # -- 5. Serialization --

    save_processed_data(
        output_dir=PROCESSED_DATA_DIR,
        train_data=train_segments['data'],
        train_labels=train_segments['labels'],
        val_data=val_segments['data'],
        val_labels=val_segments['labels'],
        test_data=test_segments['data'],
        test_labels=test_segments['labels'],
        scaler=scaler,
        metadata_df=metadata_df
    )

    print("Pipeline completed successfully.")
    
if __name__ == "__main__":
    main()
