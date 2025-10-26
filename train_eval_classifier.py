import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.trainers.data_loader import get_dataloaders
import numpy as np

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

# model = LSTMClassifier(input_size=9)
model = CNN1D(in_channels=9)
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
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

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
    y_true, y_pred = evaluate(model, val_loader)
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
y_true, y_pred = evaluate(model, test_loader)
print(classification_report(y_true, y_pred, digits=4))
print(confusion_matrix(y_true, y_pred))
