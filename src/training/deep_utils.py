import copy
import random

import numpy as np
import torch
import torch.nn as nn

from src.training.evaluation import compute_binary_metrics
from src.training.thresholds import find_best_f1_threshold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

@torch.no_grad()
def predict_classifier_scores(model, loader, device):
    model.eval()
    scores = []

    for X, _ in loader:
        X = X.to(device)
        logits = model(X).squeeze(-1)
        batch_scores = torch.sigmoid(logits)
        scores.append(batch_scores.cpu().numpy())

    return np.concatenate(scores)


@torch.no_grad()
def predict_reconstruction_scores(model, loader, device):
    model.eval()
    scores = []

    for X, _ in loader:
        X = X.to(device)
        recon = model(X)
        batch_scores = ((recon - X) ** 2).mean(dim=(1, 2))
        scores.append(batch_scores.cpu().numpy())

    return np.concatenate(scores)


def save_checkpoint(path, model, config, input_shape):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "input_shape": list(input_shape),
        },
        path,
    )


def train_classifier( model,
    train_loader, val_loader, y_val,
    *,
    epochs, learning_rate, weight_decay,
    device, pos_weight, patience,
):
    model.to(device)
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_scores = predict_classifier_scores(model, val_loader, device)
        threshold = find_best_f1_threshold(y_val, val_scores)
        val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
        val_f1 = val_metrics["f1"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_f1": val_f1,
                "val_threshold": threshold,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return history


def train_autoencoder(
    model,
    train_loader, val_loader, y_val,
    *,
    epochs, learning_rate, weight_decay,
    device, patience,
):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for X, _ in train_loader:
            X = X.to(device)

            optimizer.zero_grad()
            recon = model(X)
            loss = criterion(recon, X) # REconstruction loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_scores = predict_reconstruction_scores(model, val_loader, device)
        threshold = find_best_f1_threshold(y_val, val_scores)
        val_metrics = compute_binary_metrics(y_val, val_scores, threshold)
        val_f1 = val_metrics["f1"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_f1": val_f1,
                "val_threshold": threshold,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return history
