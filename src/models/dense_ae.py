import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    """Dense autoencoder for flattened raw windows.

    Default settings for primary benchmark: a 2 second window at 20 Hz
    with 3 accelerometer axes, giving a flattened input size of 120.
    """

    def __init__(self, input_dim: int = 120, latent_dim: int = 16, 
                 hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ---- Create Encoder block ----
        encoder_layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # --- Create decoder block ----
        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        
        x_flat = x.reshape(batch_size, -1)

        encoded = self.encoder(x_flat)
        
        decoded_flat = self.decoder(encoded)
        
        return decoded_flat.reshape(batch_size, seq_len, n_features)
