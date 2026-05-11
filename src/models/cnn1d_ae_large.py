import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding(kernel_size: int) -> int:
    return kernel_size // 2


def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = x.shape[-1]
    if current_length > target_length:
        return x[..., :target_length]
    if current_length < target_length:
        return F.pad(x, (0, target_length - current_length))
    return x


class CNN1D_AE_Large(nn.Module):
    """Larger Conv1D autoencoder for native-rate TSAD experiments.

    Expected input shape is ``(batch, seq_len, n_features)``. Four pooling
    stages create a stronger temporal bottleneck for native-rate windows such
    as ``400 x 3`` or ``476 x 3``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_sizes: tuple[int, int, int, int] = (51, 31, 21, 11),
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != 4:
            raise ValueError("kernel_sizes must contain exactly four values.")
        if any(kernel_size < 1 or kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError("kernel_sizes must be positive odd integers.")

        k1, k2, k3, k4 = kernel_sizes
        self.kernel_sizes = kernel_sizes

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=k1, padding=_same_padding(k1)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(32, 64, kernel_size=k2, padding=_same_padding(k2)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(64, 128, kernel_size=k3, padding=_same_padding(k3)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(128, 128, kernel_size=k4, padding=_same_padding(k4)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, ceil_mode=True),

            nn.Conv1d(128, 128, kernel_size=k4, padding=_same_padding(k4)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(128, 128, kernel_size=k4, padding=_same_padding(k4)),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(128, 64, kernel_size=k3, padding=_same_padding(k3)),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, kernel_size=k2, padding=_same_padding(k2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(32, in_channels, kernel_size=k1, padding=_same_padding(k1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_length = x.shape[1]
        x = x.permute(0, 2, 1)
        recon = self.decoder(self.encoder(x))
        recon = _match_length(recon, target_length)
        return recon.permute(0, 2, 1)


CNN1DAELarge = CNN1D_AE_Large
