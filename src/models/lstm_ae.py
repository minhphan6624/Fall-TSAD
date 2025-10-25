import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTM_AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        # Output layer to project decoder output back to input_dim
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)

        # Encoder forward pass; x Shape: (batch_size, sequence_length, hidden_dim)
        _, (hidden_state, cell_state) = self.encoder(x)

        # Decoder forward pass
        decoder_input = torch.zeros_like(x)  # Zero input for decoder
        decoder_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        
        output = self.output_layer(decoder_output)
        
        return output
