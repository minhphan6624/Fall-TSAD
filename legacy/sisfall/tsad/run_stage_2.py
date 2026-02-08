import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib

# Define input and output directories
IN_DIR_SPLIT = Path("data/processed/sisfall/tsad/windows")
OUT_DIR_NORM = Path("data/processed/sisfall/tsad/final")
OUT_DIR_NORM.mkdir(parents=True, exist_ok=True)

# Subject groups for splitting (based on SisFall participant IDs)
train_subjects = [f"SA{i:02d}" for i in range(1, 16)]  # SA01–SA15
val_subjects   = [f"SA{i:02d}" for i in range(16, 18)]  # SA16–SA17
test_subjects  = [f"SA{i:02d}" for i in range(18, 24)] \
                + [f"SE{i:02d}" for i in range(1, 16)]  # SA18–SA23 + SE01–SE15

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

print("Starting data splitting...")
for file_path in IN_DIR_SPLIT.glob("*.npz"):
    data = np.load(file_path, allow_pickle=True)
    meta = data["meta"].item()
    subj = meta["subject"]
    X, y = data["X"], data["y"]

    if subj in train_subjects:
        X_adl = X[y == 0]; y_adl = y[y == 0] # ADL only
        X_train.append(X_adl); y_train.append(y_adl)
    elif subj in val_subjects:
        X_adl = X[y == 0]; y_adl = y[y == 0]  # ADL only
        X_val.append(X_adl); y_val.append(y_adl)
    elif subj in test_subjects:
        X_test.append(X); y_test.append(y)

def stack(arr_list): return np.concatenate(arr_list, axis=0) if len(arr_list) else np.array([])

X_train, y_train = stack(X_train), stack(y_train)
X_val, y_val = stack(X_val), stack(y_val)
X_test, y_test = stack(X_test), stack(y_test)

print("Data splitting complete.")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

assert np.all(y_train == 0), "Found fall windows in training set!"
assert np.all(y_val == 0), "Found fall windows in validation set!"

print("Starting data normalization...")
# Initialize a list of scalers, one for each axis
scalers = [RobustScaler() for _ in range(X_train.shape[-1])]

# Fit each scaler on its corresponding axis in the training data
X_train_norm = np.zeros_like(X_train)
for i in range(X_train.shape[-1]):
    X_train_norm[:, :, i] = scalers[i].fit_transform(X_train[:, :, i].reshape(-1, 1)).reshape(X_train.shape[0], X_train.shape[1])

# Normalize a dataset using the fitted scalers
def normalize_dataset(X, fitted_scalers):
    original_shape = X.shape
    X_scaled = np.zeros_like(X)
    for i in range(original_shape[-1]):
        X_scaled[:, :, i] = fitted_scalers[i].transform(X[:, :, i].reshape(-1, 1)).reshape(original_shape[0], original_shape[1])
    return X_scaled

X_val_norm = normalize_dataset(X_val, scalers)
X_test_norm = normalize_dataset(X_test, scalers)

print("Post-normalization check:")
print("Train mean per channel:", X_train_norm.mean(axis=(0, 1)))
print("Train std per channel:", X_train_norm.std(axis=(0, 1)))

# Save normalized datasets
np.savez_compressed(OUT_DIR_NORM / "train.npz", X=X_train_norm, y=y_train, subjects=train_subjects)
np.savez_compressed(OUT_DIR_NORM / "val.npz",   X=X_val_norm,   y=y_val,   subjects=val_subjects)
np.savez_compressed(OUT_DIR_NORM / "test.npz",  X=X_test_norm,  y=y_test,  subjects=test_subjects)

# Save normalization parameters
scaler_path = OUT_DIR_NORM / "scalers.save"
joblib.dump(scalers, scaler_path)

print("TSAD dataset preprocessing (split and normalize) completed and saved to:", OUT_DIR_NORM)
