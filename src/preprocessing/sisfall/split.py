import numpy as np
from pathlib import Path

IN_DIR = Path("data/processed/sisfall/windows")
OUT_DIR = Path("data/processed/sisfall/final_tsad")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Subject groups (based on SisFall participant IDs)
train_subjects = [f"SA{i:02d}" for i in range(1, 16)]  # SA01–SA15
val_subjects   = [f"SA{i:02d}" for i in range(16, 18)] # SA16–SA17
test_subjects  = [f"SA{i:02d}" for i in range(18, 24)]  \
                + [f"SE{i:02d}" for i in range(1, 16)]  # SA18–SA23 + SE01–SE15

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for file_path in IN_DIR.glob("*.npz"):
    data = np.load(file_path, allow_pickle=True)
    meta = data["meta"].item()
    subj = meta["subject"]
    X,y = data["X"], data["y"]

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

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

assert np.all(y_train == 0), "Found fall windows in training set!"
assert np.all(y_val == 0), "Found fall windows in validation set!"

np.savez_compressed(OUT_DIR / "train.npz", X=X_train, y=y_train, subjects=train_subjects)
np.savez_compressed(OUT_DIR / "val.npz",   X=X_val,   y=y_val,   subjects=val_subjects)
np.savez_compressed(OUT_DIR / "test.npz",  X=X_test,  y=y_test,  subjects=test_subjects)

print("Train windows:", len(y_train))
print("Val windows:", len(y_val))
print("Test windows:", len(y_test))
print("Falls in test:", np.sum(y_test))
print("Fall ratio (test):", np.mean(y_test))

print("TSAD dataset split completed and saved to:", OUT_DIR)
