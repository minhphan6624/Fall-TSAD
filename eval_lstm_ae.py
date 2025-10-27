import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve,
    f1_score
)
import matplotlib.pyplot as plt

from src.models.lstm_ae import LSTM_AE
from src.trainers.data_loader import get_dataloaders

# ---- Configuration ----
DATA_DIR = Path("data/processed/sisfall/tsad/final")
MODEL_PATH = Path("runs/lstm_ae/best.pt")
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 90  # Percentile for thresholding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# --- Model config ---
INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2

# --- Load Model ---
model = LSTM_AE(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT)
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

# --- Calculate Reconstruction Errors ---
_, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

val_errors, y_val = cal_recon_errors(model, val_loader, DEVICE)
test_errors, y_test = cal_recon_errors(model, test_loader, DEVICE)

# --- Determine Threshold ---
threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
print(f"Chosen threshold (90th percentile): {threshold:.6f}")
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
precision, recall, _ = precision_recall_curve(y_test, test_errors)
print(f"ROC AUC (using continuous errors): {auc:.4f}")
print(f"Precision-Recall AUC: {np.trapezoid(precision, recall):.4f}")

output_dir = Path("runs/lstm_ae")
out_filename = output_dir / "evaluation_report.txt"
with open(out_filename, "w") as f:
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

# ROC Curve
plt.figure()
fpr, tpr, _ = roc_curve(y_test, test_errors)
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig(output_dir / "roc_curve.png", dpi=200)

# --- Plot Reconstruction Examples ---
def plot_reconstruction_examples(model, dataloader, device, output_dir, num_examples=5):
    model.eval()
    with torch.no_grad():
        # Get some samples from the test set
        data_iter = iter(dataloader)
        X_samples, y_samples = [], []
        for _ in range(num_examples):
            X, y = next(data_iter)
            X_samples.append(X[0])
            y_samples.append(y[0])
        
        X_samples = torch.stack(X_samples).to(device)
        recons_samples = model(X_samples)

        X_samples = X_samples.cpu().numpy()
        recons_samples = recons_samples.cpu().numpy()
        y_samples = np.array(y_samples)

        plt.figure(figsize=(15, 5 * num_examples))
        for i in range(num_examples):
            plt.subplot(num_examples, 1, i + 1)
            plt.plot(X_samples[i, :, 0], label='Original AccX')
            plt.plot(recons_samples[i, :, 0], label='Recon AccX', linestyle='--')
            plt.title(f"Sample {i+1} (Label: {y_samples[i]})")
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "reconstruction_examples.png", dpi=200)
        print(f"Reconstruction examples saved to {output_dir / 'reconstruction_examples.png'}")

# --- Plot Reconstruction Error Distribution ---
def plot_error_distribution(errors, labels, output_dir):
    plt.figure(figsize=(10, 6))
    
    adl_errors = errors[labels == 0]
    fall_errors = errors[labels == 1]

    plt.hist(adl_errors, bins=50, alpha=0.5, label='ADL Errors', color='blue')
    plt.hist(fall_errors, bins=50, alpha=0.5, label='Fall Errors', color='red')
    
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "error_distribution.png", dpi=200)
    print(f"Error distribution plot saved to {output_dir / 'error_distribution.png'}")

# Call the new plotting functions
plot_reconstruction_examples(model, test_loader, DEVICE, output_dir)
plot_error_distribution(test_errors, y_test, output_dir)
