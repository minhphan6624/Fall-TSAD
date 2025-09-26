from pathlib import Path
import numpy as np
import pandas as pd

ACC_1_SLICE = slice(0, 3)  # x, y, z
GYR_SLICE = slice(3, 6)  # x, y, z
ACC_2_SLICE = slice(6, 9)  # x, y, z

def load_sisfall_file(file_path, selected_cols=None):
    """
    Read a SisFall data file into a numpy array. 
    Currently supports selecting accelerometer and gyroscope data.
    """
    arr = pd.read_csv(file_path, sep=",", header=None).to_numpy(dtype=np.float32)

    if arr.shape[1] < 6:
        raise ValueError(f"Data file {file_path} has less than 6 columns.")
    
    # if selected_cols is None:
    #     out_data= np.r_[ACC_1_SLICE, GYR_SLICE]
    # else:
    #     out_data = arr[:, selected_cols]

    output = arr[:, np.r_[ACC_1_SLICE, GYR_SLICE]]

    return output
