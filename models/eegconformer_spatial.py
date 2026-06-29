# models/eegconformer_spatial.py
# Optional EEGConformer with a lightweight spatial filter.
# Use this only after the standard EEGConformer runs correctly.

import torch
from torch import nn, Tensor
from models.eegconformer import EEGConformer


class PointwiseSpatialFilter(nn.Module):
    """
    Simple cross-channel spatial filter.

    Input:  (B, C, T)
    Output: (B, n_spatial_filters, T)

    It treats EEG channels as feature channels and uses 1x1 conv across channels.
    This is intentionally simpler than the uploaded experimental spatial-filter file
    to make it stable inside the benchmark project.
    """

    def __init__(self, n_channels, n_spatial_filters=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, n_spatial_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(n_spatial_filters),
            nn.ELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B,C,T), got {tuple(x.shape)}")
        # (B, C, T) -> (B, C, 1, T)
        x = x.unsqueeze(2)
        x = self.net(x)
        # (B, F, 1, T) -> (B, F, T)
        return x.squeeze(2)


class EEGConformerWithSpatialFilter(nn.Module):
    def __init__(
        self,
        n_outputs,
        n_chans,
        n_times,
        n_spatial_filters=32,
        spatial_dropout=0.1,
        **conformer_kwargs,
    ):
        super().__init__()
        self.spatial_filter = PointwiseSpatialFilter(
            n_channels=n_chans,
            n_spatial_filters=n_spatial_filters,
            dropout=spatial_dropout,
        )
        self.conformer = EEGConformer(
            n_outputs=n_outputs,
            n_chans=n_spatial_filters,
            n_times=n_times,
            **conformer_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_filter(x)
        return self.conformer(x)
