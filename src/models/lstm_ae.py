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

        def encode(self, x):
            _, (hidden_state, _) = self.encoder(x)
            return hidden_state[-1]  # shape [batch, hidden_dim]


        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim, # Changed to hidden_dim
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layer to project decoder output back to n_features
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)

        # Encoder forward pass
        # Shape: (batch_size, sequence_length, hidden_dim)
        _, (hidden_state, cell_state) = self.encoder(x)


        decoder_input = torch.zeros_like(x)  # [batch, seq_len, n_features]
        decoder_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        
        output = self.output_layer(decoder_output)
        
        # Project decoder output to original feature dimension
        output = self.output_layer(decoder_output)
        
        return output
