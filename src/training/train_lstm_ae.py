import argparse
import json

import pandas as pd

from src.models.lstm_ae import LSTM_AE
from src.training.data import label_counts, load_window_data, make_dataloaders
from src.training.deep_utils import (
    get_device,
    predict_reconstruction_scores,
    save_checkpoint,
    set_seed,
    train_autoencoder,
)
from src.training.evaluation import compute_binary_metrics
from src.training.run_utils import make_run_dir, save_json, save_predictions
from src.training.thresholds import find_best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM autoencoder TSAD model.")
    parser.add_argument("--dataset", default="sisfall")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--run-root", default="runs/benchmark")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.model_seed)

    data = load_window_data(args.dataset, "tsad", data_root=args.data_root)
    train_loader, val_loader, test_loader = make_dataloaders(
        data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    seq_len, n_channels = data["train"]["X"].shape[1:]
    device = get_device(args.device)
    model = LSTM_AE(
        input_dim=n_channels,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=seq_len,
    )

    history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        y_val,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        patience=args.patience,
    )

    val_scores = predict_reconstruction_scores(model, val_loader, device)
    test_scores = predict_reconstruction_scores(model, test_loader, device)

    threshold = find_best_f1_threshold(y_val, val_scores)
    val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
    test_metrics = compute_binary_metrics(y_test, test_scores, threshold)

    run_dir = make_run_dir(
        args.run_root, args.dataset, "tsad", "lstm_ae", args.model_seed
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", vars(args))
    save_json(
        run_dir / "metrics.json",
        {
            "train_label_counts": label_counts(y_train),
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    save_predictions(
        run_dir / "predictions_val.csv",
        data["val"]["meta"],
        y_val,
        val_scores,
        threshold,
    )
    save_predictions(
        run_dir / "predictions_test.csv",
        data["test"]["meta"],
        y_test,
        test_scores,
        threshold,
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    save_checkpoint(run_dir / "model.pt", model, vars(args), data["train"]["X"].shape[1:])

    print(f"Saved LSTM autoencoder run to {run_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
