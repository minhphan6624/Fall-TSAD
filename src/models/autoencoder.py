import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)
        # Flatten the input
        batch_size, seq_len, n_features = x.shape
        x_flat = x.view(batch_size, -1) # -> (batch_size, sequence_length * n_features)

        encoded = self.encoder(x_flat) # Encode to latent space (batch_size, latent_dim)

        decoded_flat = self.decoder(encoded) # Decode back to original dimension (batch_size, sequence_length * n_features)
        decoded = decoded_flat.view(batch_size, seq_len, n_features) # Reshape
        return decoded
