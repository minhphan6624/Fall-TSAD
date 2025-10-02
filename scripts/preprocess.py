import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the src directory to the Python path to import custom modules
# This might need adjustment based on the final project structure and how it's run
if str(Path('../src').resolve()) not in sys.path:
    sys.path.insert(0, str(Path('../src').resolve()))

from src.datasets.sisfall_paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.datasets.sisfall_metadata import build_metadata
from src.datasets.sisfall_split import split_data_by_subject
from src.datasets.sisfall_loader import load_sisfall_file
from src.datasets.sisfall_normalize import StandardScaler
from src.datasets.sisfall_segment import segment_dataset
from src.datasets.sisfall_serialize import save_processed_data, load_processed_data


def run_preprocessing():
    # Create processed data directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data will be saved to: {PROCESSED_DATA_DIR}")

    ## Configuration Parameters
    WINDOW_SIZE = 600  # 3 seconds at 200Hz
    OVERLAP = 300      # 50% overlap
    RANDOM_STATE = 42  # For reproducibility of subject split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    # Test ratio will be 1 - TRAIN_RATIO - VAL_RATIO

    ## Step 1: Metadata Extraction and Subject-wise Splitting
    print("\n--- Step 1: Building Metadata and Splitting by Subject ---")
    metadata_df = build_metadata(RAW_DATA_DIR)
    print(f"Total files found: {len(metadata_df)}")
    print(f"Unique subjects: {metadata_df['subject'].nunique()}")

    train_meta, val_meta, test_meta = split_data_by_subject(
        metadata_df, 
        train_ratio=TRAIN_RATIO, 
        val_ratio=VAL_RATIO, 
        random_state=RANDOM_STATE
    )

    print(f"Train subjects: {train_meta['subject'].nunique()} (files: {len(train_meta)})")
    print(f"Validation subjects: {val_meta['subject'].nunique()} (files: {len(val_meta)})")
    print(f"Test subjects: {test_meta['subject'].nunique()} (files: {len(test_meta)})")

    ## Step 2: Load Data for Each Split with Unit Conversion
    print("\n--- Step 2: Loading Raw Data with Unit Conversion ---")

    def load_data_for_split(meta_df: pd.DataFrame) -> list[np.ndarray]:
        data_list = []
        for idx, row in meta_df.iterrows():
            file_path = Path(row['path'])
            try:
                data = load_sisfall_file(file_path)
                data_list.append(data)
            except ValueError as e:
                print(f"Skipping {file_path} due to error: {e}")
        return data_list

    train_raw_data = load_data_for_split(train_meta)
    val_raw_data = load_data_for_split(val_meta)
    test_raw_data = load_data_for_split(test_meta)

    print(f"Loaded {len(train_raw_data)} training data files.")
    print(f"Loaded {len(val_raw_data)} validation data files.")
    print(f"Loaded {len(test_raw_data)} test data files.")

    if train_raw_data:
        print(f"Example train data shape (first file): {train_raw_data[0].shape}")

    ## Step 3: Normalization (Standardization)
    print("\n--- Step 3: Normalizing Data ---")

    # Concatenate all training data files to fit the scaler
    full_train_data_for_scaler = np.concatenate(train_raw_data, axis=0)

    scaler = StandardScaler()
    scaler.fit(full_train_data_for_scaler)

    print(f"Scaler fitted. Mean: {scaler.mean}, Std: {scaler.std}")

    # Apply the transformation to each individual data file in all splits
    train_norm_data = [scaler.transform(d) for d in train_raw_data]
    val_norm_data = [scaler.transform(d) for d in val_raw_data]
    test_norm_data = [scaler.transform(d) for d in test_raw_data]

    ## Step 4: Segmentation
    print("\n--- Step 4: Segmenting Data ---")

    # Extract labels corresponding to the order of data files in each split
    train_labels_list = train_meta['is_fall'].tolist()
    val_labels_list = val_meta['is_fall'].tolist()
    test_labels_list = test_meta['is_fall'].tolist()

    train_X, train_y = segment_dataset(train_norm_data, train_labels_list, WINDOW_SIZE, OVERLAP)
    val_X, val_y = segment_dataset(val_norm_data, val_labels_list, WINDOW_SIZE, OVERLAP)
    test_X, test_y = segment_dataset(test_norm_data, test_labels_list, WINDOW_SIZE, OVERLAP)

    print(f"Segmented Training Data Shape: {train_X.shape}, Labels Shape: {train_y.shape}")
    print(f"Segmented Validation Data Shape: {val_X.shape}, Labels Shape: {val_y.shape}")
    print(f"Segmented Test Data Shape: {test_X.shape}, Labels Shape: {test_y.shape}")

    ## Step 5: Serialization
    print("\n--- Step 5: Saving Processed Data ---")

    save_processed_data(
        output_dir=PROCESSED_DATA_DIR,
        train_data=train_X, train_labels=train_y,
        val_data=val_X, val_labels=val_y,
        test_data=test_X, test_labels=test_y,
        scaler=scaler,
        metadata_df=metadata_df
    )

    print("Data preprocessing pipeline completed and data saved.")

if __name__ == "__main__":
    run_preprocessing()
