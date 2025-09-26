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

        # Encoder
        encoder_output, (hidden_state, cell_state) = self.encoder(x)
        # Prepare decoder input (repeat the last hidden state for each time step)
        decoder_input = encoder_output[:, -1, :].unsqueeze(1).repeat(1, x.shape[1], 1)
        
        decoder_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        
        return decoder_output
    

# Decoder - input to decoder is the last encoder output, repeated for sequence length
# We need to repeat the last hidden state for the sequence length
# The input to the decoder should be the hidden state from the encoder
# For an autoencoder, the decoder tries to reconstruct the input sequence.
# A common approach is to feed the last hidden state of the encoder as the initial hidden state of the decoder,
# and then feed a sequence of zeros or the last encoder output repeated.
# Here, we'll use the last encoder output repeated for simplicity.

# Take the last output of the encoder and repeat it for the sequence length
# encoder_output[:, -1, :] gives the last output for all batches
# We need to expand its dimensions to match the input shape expected by the decoder
# x.shape[1] is the sequence length
