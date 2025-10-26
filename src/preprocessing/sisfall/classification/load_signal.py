from pathlib import Path
import numpy as np
import pandas as pd

# -- Sensor characteristics from Readme.txt --
# ADXL345 (Accelerometer 1)
ADXL345_RESOLUTION = 13
ADXL345_RANGE = 16  # +-16g

# ITG3200 (Gyroscope)
ITG3200_RESOLUTION = 16
ITG3200_RANGE = 2000  # +-2000 deg/s

# MMA8451Q (Accelerometer 2)
MMA8451Q_RESOLUTION = 14    
MMA8451Q_RANGE = 8  # +-8g

# -- Conversion formulas --
# Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
# Angular velocity [deg/s]: [(2*Range)/(2^Resolution)]*RD
ACC_1_CONVERSION_FACTOR = (2 * ADXL345_RANGE) / (2**ADXL345_RESOLUTION)
GYR_CONVERSION_FACTOR = (2 * ITG3200_RANGE) / (2**ITG3200_RESOLUTION)
ACC_2_CONVERSION_FACTOR = (2* MMA8451Q_RANGE) / (2**MMA8451Q_RESOLUTION)

ACC_1_SLICE = slice(0,3)
GYR_SLICE = slice(3,6)
ACC_2_SLICE = slice(6,9)

def load_signal(file_path: Path):
    df = pd.read_csv(file_path, sep=",", header=None)

    # The last column might contain a trailing semicolon, remove it
    if df.iloc[:, -1].dtype == 'object':
        df.iloc[:, -1] = df.iloc[:, -1].str.replace(';','', regex=False)

    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[1] < 9:
        raise ValueError(f"Data file {file_path} has less than 9 columns")
    
    # Slice Data
    acc_1_data = arr[:, ACC_1_SLICE]
    acc_2_data = arr[:, ACC_2_SLICE]
    gyr_data = arr[:, GYR_SLICE]

    # Apply unit conversion
    acc_1_data_g = acc_1_data * ACC_1_CONVERSION_FACTOR
    acc_2_data_g = acc_2_data * ACC_2_CONVERSION_FACTOR
    gyr_data_degs = gyr_data * GYR_CONVERSION_FACTOR

    return {
        "acc1": acc_1_data_g,
        "acc2": acc_2_data_g,
        "gyr": gyr_data_degs
    }

