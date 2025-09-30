import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers, dropout):
        super(LSTM_AE, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)

        # Encoder forward pass
        # Shape: (batch_size, sequence_length, hidden_dim)
        encoder_output, (hidden_state, cell_state) = self.encoder(x)

        # Prepare decoder input (repeat the last hidden state for each time step)
        # Extract last time step's output -> add dimension for sequence length -> repeat for sequence length
        decoder_input = encoder_output[:, -1, :].unsqueeze(1).repeat(1, x.shape[1], 1)
        
        # Decoder forward pass
        decoder_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        
        return decoder_output
