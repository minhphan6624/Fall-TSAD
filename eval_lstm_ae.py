import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

from src.models.lstm_ae import LSTM_AE

# ---- Configuration ----

DATA_DIR = Path("data/processed/sisfall/ready")
MODEL_PATH = Path("runs/lstm_ae/best.pt")
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 90  # Percentile for thresholding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# --- Load Model ---
model = LSTM_AE(n_features=6, hidden_dim=64, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Helper ----
def cal_recon_errors(model, loader, device):
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            recon = model(X)

            batch_errors = torch.mean((recon - X)**2, dim=(1,2)).cpu().numpy() # MSE per sample

            errors.extend(batch_errors)
            labels.extend(y.cpu().numpy())
    return np.array(errors), np.array(labels)

# --- Load Data ---
# val_data = np.load(DATA_DIR / "val.npz", allow_pickle=True)
# test_data = np.load(DATA_DIR / "test.npz", allow_pickle=True)
# X_val, y_val = val_data["X"], val_data["y"]
# X_test, y_test = test_data["X"], test_data["y"]

# --- Get dataloaders ---
from src.trainers.data_loader import get_dataloaders
_, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

# --- Calculate Reconstruction Errors ---
val_errors, y_val = cal_recon_errors(model, val_loader, DEVICE)
test_errors, y_test = cal_recon_errors(model, test_loader, DEVICE)

# --- Determine Threshold ---
threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
print(f"Chosen threshold (95th percentile): {threshold:.6f}")

y_pred = (test_errors > threshold).astype(int)

print(f"\n--- Evaluation Results ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))

auc = roc_auc_score(y_test, test_errors)
print(f"ROC AUC (using continuous errors): {auc:.4f}")

# ---  Visualize distributions ---
plt.figure(figsize=(8,4))
plt.hist(test_errors[y_test==0], bins=50, alpha=0.6, label="ADL")
plt.hist(test_errors[y_test==1], bins=50, alpha=0.6, label="Fall")
plt.axvline(threshold, color='red', linestyle='--', label='Threshold (95%)')
plt.legend()
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution (Test Set)")
plt.tight_layout()
plt.savefig("runs/lstm_ae/error_distribution.png", dpi=200)