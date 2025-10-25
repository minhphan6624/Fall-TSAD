import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt

from src.models.lstm_ae import LSTM_AE

# ---- Configuration ----

DATA_DIR = Path("data/processed/sisfall/ready")
MODEL_PATH = Path("runs/lstm_ae/best.pt")
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 80  # Percentile for thresholding

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

            # batch_errors = torch.mean((recon - X)**2, dim=(1,2)).cpu().numpy() # MSE per sample
            batch_errors = torch.mean(torch.abs(recon - X), dim=(1,2)).cpu().numpy() # MAE per sample

            errors.extend(batch_errors)
            labels.extend(y.cpu().numpy())
    return np.array(errors), np.array(labels)

# --- Get dataloaders ---
from src.trainers.data_loader import get_dataloaders
_, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

# --- Calculate Reconstruction Errors ---
val_errors, y_val = cal_recon_errors(model, val_loader, DEVICE)
test_errors, y_test = cal_recon_errors(model, test_loader, DEVICE)

# --- Determine Threshold ---
threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
print(f"Chosen threshold (85th percentile): {threshold:.6f}")
y_pred = (test_errors > threshold).astype(int)

# --- Tune threshold based on F1 score (optional) ---

# thresholds = np.linspace(min(val_errors), max(val_errors), 100)
# best_f1, best_t = 0, 0

# for t in thresholds:
#     preds = (val_errors > t).astype(int)
#     f1 = f1_score(y_val, preds)
#     if f1 > best_f1:
#         best_f1, best_t = f1, t
# print("Best threshold:", best_t, "F1:", best_f1)

# y_pred = (test_errors > best_t).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))

auc = roc_auc_score(y_test, test_errors)
print(f"ROC AUC (using continuous errors): {auc:.4f}")

precision, recall, _ = precision_recall_curve(y_test, test_errors)
print(f"Precision-Recall AUC: {np.trapezoid(precision, recall):.4f}")

with open("runs/lstm_ae/evaluation_report.txt", "w") as f:
    f.write(f"--- Evaluation Results ---\n")
    f.write(f"{confusion_matrix(y_test, y_pred)}\n")
    f.write(f"{classification_report(y_test, y_pred, target_names=['ADL', 'Fall'])}\n")
    f.write(f"ROC AUC (using continuous errors): {auc:.4f}\n")
    f.write(f"Precision-Recall AUC: {np.trapezoid(precision, recall):.4f}\n")

# Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("runs/lstm_ae/precision_recall_curve.png", dpi=200) 
