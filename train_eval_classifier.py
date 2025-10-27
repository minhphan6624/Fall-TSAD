import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve
)
from src.trainers.data_loader import get_dataloaders
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "data/processed/sisfall/classification/final"
RUN_DIR = Path("runs/cnn1d_classifier")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data loading ---
train_loader, val_loader, test_loader = get_dataloaders(processed_dir=Path(DATA_DIR), batch_size=BATCH_SIZE)

# --- Choose model ---
from src.models.lstm_classifier import LSTMClassifier
from src.models.cnn1d import CNN1D

model = LSTMClassifier(input_size=3)
# model = CNN1D(in_channels=3)
model.to(DEVICE)

# --- Handle imbalance with weighted loss ---
# train_ds = train_loader.dataset
# y_train = train_ds.y 
# n_fall = np.sum(y_train == 1)
# n_adl  = np.sum(y_train == 0)
# weights = torch.tensor([1.0, n_adl / n_fall], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=None)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# --- Helper function to evaluate the model ---
def evaluate(model, dataloader):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy()) 
    return np.array(y_true), np.array(y_pred), np.array(y_score)

# --- Training loop ---
best_val_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    print("\n=== TRAINING EPOCH %d ===" % epoch)

    for X, y in train_loader:

        # --- Training step ---
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # --- Validation ---
    y_true, y_pred, _ = evaluate(model, val_loader) # y_score is not used in validation
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    val_f1 = report["1"]["f1-score"]
    val_acc = report["accuracy"]

    scheduler.step(1 - val_f1)
    print(f"Epoch {epoch:02d}: Train loss={train_loss:.4f}, acc={train_acc:.3f}, Val acc={val_acc:.3f}, F1={val_f1:.3f}")

    # --- Save best model ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), RUN_DIR / "best_model.pt")
        print("New best model saved.")

# --- Final evaluation on test set ---
print("\n=== TEST EVALUATION ===")
model.load_state_dict(torch.load(RUN_DIR / "best_model.pt"))
y_true, y_pred, y_score = evaluate(model, test_loader)

# Print and save classification report and confusion matrix
report_str = classification_report(y_true, y_pred, digits=4, target_names=["ADL", "Fall"])
cm_str = str(confusion_matrix(y_true, y_pred))
print(report_str)
print(cm_str)

# Calculate AUC scores
roc_auc = roc_auc_score(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_auc = np.trapezoid(precision, recall)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Save evaluation report to file
output_report_path = RUN_DIR / "evaluation_report.txt"
with open(output_report_path, "w") as f:
    f.write("--- Evaluation Results ---\n")
    f.write(f"Confusion Matrix:\n{cm_str}\n\n")
    f.write(f"Classification Report:\n{report_str}\n\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n")
print(f"Evaluation report saved to {output_report_path}")

# Plot and save Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
pr_curve_path = RUN_DIR / "precision_recall_curve.png"
plt.savefig(pr_curve_path, dpi=200)
print(f"Precision-Recall curve saved to {pr_curve_path}")

# Plot and save ROC Curve
plt.figure()
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
roc_curve_path = RUN_DIR / "roc_curve.png"
plt.savefig(roc_curve_path, dpi=200)
print(f"ROC curve saved to {roc_curve_path}")
