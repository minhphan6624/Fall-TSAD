import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler

IN_DIR = Path("data/processed/sisfall/final_tsad")
OUT_DIR = Path("data/processed/sisfall/ready")

# Input file paths
train_dir = IN_DIR / "train.npz"
val_dir = IN_DIR / "val.npz"
test_dir = IN_DIR / "test.npz"

train_data = np.load(IN_DIR / "train.npz", allow_pickle=True)
val_data = np.load(IN_DIR / "val.npz", allow_pickle=True)
test_data = np.load(IN_DIR / "test.npz", allow_pickle=True)

X_train = train_data["X"]
X_val = val_data["X"]
X_test = test_data["X"]

# # Calculate mean and std over training set
# mean_train = np.mean(X_train, axis=(0,1)) # Shape (num_features, )
# std_train = np.std(X_train, axis=(0,1)) # Shape (num_features, )

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

# Save normalized datasets
np.savez_compressed(OUT_DIR / "train.npz", X=X_train_norm, y=train_data["y"])
np.savez_compressed(OUT_DIR / "val.npz",   X=X_val_norm,   y=val_data["y"])
np.savez_compressed(OUT_DIR / "test.npz",  X=X_test_norm,  y=test_data["y"])

# Save the RobustScaler for future use
import joblib
scaler_path = OUT_DIR / "robust_scaler.save"
joblib.dump(scaler, scaler_path)
print("Normalization complete. Scaler saved to:", scaler_path)