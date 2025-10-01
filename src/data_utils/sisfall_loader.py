from pathlib import Path
import numpy as np
import pandas as pd

# Sensor characteristics from Readme.txt
# ADXL345 (Accelerometer 1)
ADXL345_RESOLUTION = 13
ADXL345_RANGE = 16  # +-16g

# ITG3200 (Gyroscope)
ITG3200_RESOLUTION = 16
ITG3200_RANGE = 2000  # +-2000 deg/s

# Conversion formulas:
# Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
# Angular velocity [deg/s]: [(2*Range)/(2^Resolution)]*RD

ADXL345_CONVERSION_FACTOR = (2 * ADXL345_RANGE) / (2**ADXL345_RESOLUTION)
ITG3200_CONVERSION_FACTOR = (2 * ITG3200_RANGE) / (2**ITG3200_RESOLUTION)


ACC_1_SLICE = slice(0, 3)  # x, y, z for ADXL345
GYR_SLICE = slice(3, 6)  # x, y, z for ITG3200
ACC_2_SLICE = slice(6, 9)  # x, y, z for MMA8451Q (not used in first 6 columns)

def load_sisfall_file(file_path: Path) -> np.ndarray:
    """
    Read a SisFall data file into a numpy array and convert raw bit values
    to physical units (g for accelerometer, deg/s for gyroscope).
    Extracts the first 6 columns (ADXL345 accelerometer and ITG3200 gyroscope).
    """
    df = pd.read_csv(file_path, sep=",", header=None)

    # The last column might contain a trailing semicolon. Remove it.
    # Check if the last column is of object type (string) before applying .str accessor
    if df.iloc[:, -1].dtype == 'object':
        df.iloc[:, -1] = df.iloc[:, -1].str.replace(';', '', regex=False)
    
    arr = df.to_numpy(dtype=np.float32)

    if arr.shape[1] < 6:
        raise ValueError(f"Data file {file_path} has less than 6 columns.")
    
    # Extract the first 6 columns
    accel_data = arr[:, ACC_1_SLICE]
    gyro_data = arr[:, GYR_SLICE]

    # Apply unit conversion
    accel_data_g = accel_data * ADXL345_CONVERSION_FACTOR
    gyro_data_degs = gyro_data * ITG3200_CONVERSION_FACTOR

    # Concatenate the converted data
    output = np.concatenate((accel_data_g, gyro_data_degs), axis=1)

    return output
