import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, 200, 2*hidden)
        out = out[:, -1, :]          # last time step
        return self.fc(out)
