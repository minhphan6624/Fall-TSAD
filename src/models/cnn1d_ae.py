import torch
import torch.nn as nn
import torch.nn.functional as F


def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = x.shape[-1]
    if current_length > target_length:
        return x[..., :target_length]
    if current_length < target_length:
        return F.pad(x, (0, target_length - current_length))
    return x


class CNN1D_AE(nn.Module):
    """Small Conv1D autoencoder for primary 20 Hz TSAD experiments.

    Expected input shape is ``(batch, seq_len, n_features)``. The architecture
    is fully convolutional, so it can reconstruct 20 Hz windows longer than the
    primary ``40 x 3`` setup as long as the input channel count is unchanged.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_length = x.shape[1]
        x = x.permute(0, 2, 1)
        recon = self.decoder(self.encoder(x))
        recon = _match_length(recon, target_length)
        return recon.permute(0, 2, 1)


CNN1DAE = CNN1D_AE
