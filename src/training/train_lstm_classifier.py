import argparse
import json

import pandas as pd
import torch

from src.models.lstm_classifier import LSTMClassifier
from src.training.data import label_counts, load_window_data, make_dataloaders
from src.training.deep_utils import (
    get_device,
    predict_classifier_scores,
    save_checkpoint,
    set_seed,
    train_classifier,
)
from src.training.evaluation import compute_binary_metrics
from src.training.run_utils import make_run_dir, save_json, save_predictions
from src.training.thresholds import find_best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM classifier.")
    parser.add_argument("--dataset", default="sisfall")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--run-root", default="runs/benchmark")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense-dropout", type=float, default=0.2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.model_seed)

    data = load_window_data(args.dataset, "classification", data_root=args.data_root)
    train_loader, val_loader, test_loader = make_dataloaders(
        data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    y_test = data["test"]["y"]

    normal_count = int((y_train == 0).sum())
    fall_count = int((y_train == 1).sum())
    pos_weight = torch.tensor([normal_count / fall_count], dtype=torch.float32)

    device = get_device(args.device)
    model = LSTMClassifier(
        input_size=data["train"]["X"].shape[2],
        hidden_size=args.hidden_size,
        dense_units=args.dense_units,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dense_dropout=args.dense_dropout,
    )

    history = train_classifier(
        model,
        train_loader,
        val_loader,
        y_val,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        pos_weight=pos_weight,
        patience=args.patience,
    )

    val_scores = predict_classifier_scores(model, val_loader, device)
    test_scores = predict_classifier_scores(model, test_loader, device)

    threshold = find_best_f1_threshold(y_val, val_scores)
    val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
    test_metrics = compute_binary_metrics(y_test, test_scores, threshold)

    run_dir = make_run_dir(
        args.run_root,
        args.dataset,
        "classification",
        "lstm_classifier",
        args.model_seed,
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

    print(f"Saved LSTM classifier run to {run_dir}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
