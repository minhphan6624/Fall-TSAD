import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt

from src.models.autoencoder import Autoencoder
from src.trainers.data_loader import get_dataloaders
from src.utils.set_seed import set_seed

# --- Helper ----
def cal_recon_errors(model, loader, device):
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            recon = model(X)

            # MAE per sample
            batch_errors = torch.mean(torch.abs(recon - X), dim=(1,2)).cpu().numpy() 

            errors.extend(batch_errors)
            labels.extend(y.cpu().numpy())
    return np.array(errors), np.array(labels)

def main(args):
    set_seed(args.seed)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    data_dir = Path(args.data_dir)
    
    # Get dataloaders (train_loader is needed for thresholding on ADL data)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, args.batch_size)

    # Determine input_dim for the Autoencoder from a sample batch
    sample_batch, _ = next(iter(train_loader))
    _, seq_len, n_features = sample_batch.shape
    input_dim = seq_len * n_features

    # --- 2. Initialize Model ---
    model = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim, hidden_dims=args.hidden_dims)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # --- 3. Calculate Reconstruction Errors ---
    # Calculate errors on training ADL data to determine a robust threshold
    train_adl_errors, _ = cal_recon_errors(model, train_loader, DEVICE)
    test_errors, y_test = cal_recon_errors(model, test_loader, DEVICE)

    # --- 4. Determine Threshold ---
    # Use only ADL data from training set to determine threshold
    threshold = np.percentile(train_adl_errors, args.threshold_percentile)
    print(f"Chosen threshold ({args.threshold_percentile}th percentile of training ADL errors): {threshold:.6f}")
    y_pred = (test_errors > threshold).astype(int)

    # --- 5. Calculate and Report Metrics ---
    print("\n--- Classification Report ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))

    auc = roc_auc_score(y_test, test_errors)
    print(f"ROC AUC (using continuous errors): {auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    pr_auc = np.trapezoid(precision, recall)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # --- 6. Save Results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_filename = output_dir / "evaluation_report.txt"
    with open(out_filename, "w") as f:
        f.write(f"--- Evaluation Results ---\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Latent Dimension: {args.latent_dim}\n")
        f.write(f"Hidden Dimensions: {args.hidden_dims}\n")
        f.write(f"Threshold Percentile: {args.threshold_percentile}\n")
        f.write(f"Chosen Threshold: {threshold:.6f}\n\n")
        f.write(f"{confusion_matrix(y_test, y_pred)}\n\n")
        f.write(f"{classification_report(y_test, y_pred, target_names=['ADL', 'Fall'])}\n")
        f.write(f"ROC AUC (using continuous errors): {auc:.4f}\n")
        f.write(f"Precision-Recall AUC: {pr_auc:.4f}\n")
    print(f"Evaluation report saved to: {out_filename}")

    # --- 7. Plot Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    pr_curve_path = output_dir / "precision_recall_curve.png"
    plt.savefig(pr_curve_path, dpi=200) 
    print(f"Precision-Recall curve saved to: {pr_curve_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Autoencoder model for anomaly detection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_dir", type=str, default="data/processed/sisfall/final_tsad",
                        help="Directory containing processed data (train.npz, val.npz, test.npz).")
    parser.add_argument("--model_path", type=str, 
                        default="runs/autoencoder/ae_latent32_lr0.001/best.pt",
                        help="Path to the trained Autoencoder model checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, default="runs/autoencoder/evaluation",
                        help="Directory to save evaluation reports and plots.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--latent_dim", type=int, default=32, help="Dimension of the latent space used in the trained model.")
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[128, 64],
                        help="Hidden layer dimensions used in the trained model. E.g., --hidden_dims 128 64")
    parser.add_argument("--threshold_percentile", type=float, default=95,
                        help="Percentile of training ADL reconstruction errors to use as threshold.")

    args = parser.parse_args()
    main(args)
