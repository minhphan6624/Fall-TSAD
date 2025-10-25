import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import joblib

# Define input and output directories
IN_DIR_SPLIT = Path("data/processed/sisfall/windows")
OUT_DIR_NORM = Path("data/processed/sisfall/final_tsad")
OUT_DIR_NORM.mkdir(parents=True, exist_ok=True)

# Subject groups for splitting (based on SisFall participant IDs)
train_subjects = [f"SA{i:02d}" for i in range(1, 16)]  # SA01–SA15
val_subjects   = [f"SA{i:02d}" for i in range(16, 18)] # SA16–SA17
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
# Apply RobustScaler on training set
scaler = RobustScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # combine windows
scaler.fit(X_train_flat)

# Normalize a dataset using the fitted scaler
def normalize_dataset(X):
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_flat).reshape(original_shape)
    return X_scaled

X_train_norm = normalize_dataset(X_train)
X_val_norm = normalize_dataset(X_val)
X_test_norm = normalize_dataset(X_test)

print("Post-normalization check:")
print("Train median per channel:", np.median(X_train_norm, axis=(0,1)))
print("Train IQR per channel:", np.percentile(X_train_norm, 75, axis=(0,1)) - np.percentile(X_train_norm, 25, axis=(0,1)))

# Save normalized datasets
np.savez_compressed(OUT_DIR_NORM / "train.npz", X=X_train_norm, y=y_train, subjects=train_subjects)
np.savez_compressed(OUT_DIR_NORM / "val.npz",   X=X_val_norm,   y=y_val,   subjects=val_subjects)
np.savez_compressed(OUT_DIR_NORM / "test.npz",  X=X_test_norm,  y=y_test,  subjects=test_subjects)

# Save the RobustScaler for future use
scaler_path = OUT_DIR_NORM / "robust_scaler.save"
joblib.dump(scaler, scaler_path)
print("Normalization complete. Scaler saved to:", scaler_path)

print("TSAD dataset preprocessing (split and normalize) completed and saved to:", OUT_DIR_NORM)
