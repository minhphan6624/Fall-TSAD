import torch
import torch.nn as nn


class LSTM_AE(nn.Module):
    """LSTM autoencoder for sequence reconstruction.

    Default settings match the primary benchmark input of ``40 x 3``.
    """

    def __init__(
        self,
        input_dim: int = 3, hidden_dim: int = 64, latent_dim: int = 32,
        num_layers: int = 1, dropout: float = 0.2, seq_len: int = 40,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )
        self.encoder_dropout = nn.Dropout(dropout)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        _, (hidden_state, _) = self.encoder(x)
        last_hidden = self.encoder_dropout(hidden_state[-1])
        latent = self.to_latent(last_hidden)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input)
        return self.output_layer(decoder_output)


LSTMAutoencoder = LSTM_AE
