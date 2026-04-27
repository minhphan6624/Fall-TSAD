import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Primary-benchmark LSTM classifier for raw accelerometer windows."""

    def __init__(
        self, input_size: int = 3, hidden_size: int = 64, 
        dense_units: int = 32, num_layers: int = 1, 
        dropout: float = 0.3, dense_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.head(last_hidden)
