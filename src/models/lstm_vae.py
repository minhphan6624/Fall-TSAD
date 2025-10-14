import torch
import torch.nn as nn

class LSTM_VAE(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, num_layers, dropout):
        super(LSTM_VAE, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Latent space mapping
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer to reconstruct the original features
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)

        # --- Encoder ---
        # encoder_output shape: (batch_size, sequence_length, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        _, (hidden, _) = self.encoder(x)
        
        # We use the hidden state from the last layer
        last_hidden = hidden[-1] # Shape: (batch_size, hidden_dim)

        # --- Latent Space ---
        mu = self.fc_mu(last_hidden)
        log_var = self.fc_log_var(last_hidden)
        
        # --- Reparameterization ---
        z = self.reparameterize(mu, log_var) # Shape: (batch_size, latent_dim)

        # --- Decoder ---
        # Prepare input for the decoder LSTM
        # We need to repeat the latent vector z for each time step of the sequence
        seq_len = x.shape[1]
        decoder_hidden = self.decoder_input(z) # Shape: (batch_size, hidden_dim)
        
        # The decoder needs an initial hidden state. We can use the transformed latent vector.
        # We need to unsqueeze it to match the (num_layers, batch_size, hidden_dim) shape.
        decoder_hidden_state = decoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Repeat the decoder hidden state for each time step
        decoder_input_seq = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decoder forward pass
        decoder_output, _ = self.decoder(decoder_input_seq, (decoder_hidden_state, decoder_hidden_state)) # Using same for hidden and cell state
        
        # --- Reconstruction ---
        reconstruction = self.output_layer(decoder_output) # Shape: (batch_size, sequence_length, n_features)
        
        return reconstruction, mu, log_var
