import numpy as np
import torch
from src.models.lstm_ae import LSTM_AE

model = LSTM_AE(n_features=6, hidden_dim=64, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load("runs/lstm_ae/best.pt"))
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

test = np.load("data/processed/sisfall/ready/test.npz")
X_test = torch.tensor(test["X"], dtype=torch.float32)
y_test = test["y"]

with torch.no_grad():
    recon = model(X_test)
    errors = torch.mean((recon - X_test)**2, dim=(1,2)).numpy() # Mean squared error per sample
    # errors = np.mean(np.abs(recon - X_data), axis=(1,2))

print("Average reconstruction error (ADL vs falls):")
print("ADL:", errors[y_test==0].mean())
print("Falls:", errors[y_test==1].mean())
