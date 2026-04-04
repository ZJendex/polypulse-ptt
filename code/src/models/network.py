"""Single-site U-Net pulse detection network with spatial reduction and bidirectional LSTM bottleneck.
Input: (N, L, C) radar window -> Output: (N, L, 1) per-timestep pulse probability.
"""

import torch
import torch.nn as nn
from typing import List


class SpatialReductionPreprocessing(nn.Module):
    """2-D conv + top-k selection to reduce spatial (range-angle bin) dimension."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.spatial_temporal_processing = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.channel_adjust = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # (N, spatial, temporal) -> (N, out_channels, temporal)
        x = x.unsqueeze(1)  # (N, 1, spatial, temporal)
        x = self.spatial_temporal_processing(x)
        x = torch.topk(x, self.out_channels, dim=2)[0]
        x = self.channel_adjust(x)
        return x.squeeze(1)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        features = self.conv(x)
        return self.pool(features), features  # (pooled, skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.transposed_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PulseDetectionNet(nn.Module):
    """U-Net with spatial reduction -> encoder -> biLSTM bottleneck -> decoder -> sigmoid.

    encode/bottleneck/decode are split so MultiSitePulseDetectionNet can
    inject cross-site attention between bottleneck and decoder.
    """

    def __init__(self,
                 seq_len: int,
                 in_channels: int,  # kept for config compat; spatial reduction replaces it
                 reduce_channels: int = 8,
                 hidden_channels: List[int] = [32, 64, 128],
                 kernel_size: int = 7,
                 use_lstm: bool = True,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.use_lstm = use_lstm

        self.spatial_processing = SpatialReductionPreprocessing(reduce_channels)
        current_channels = reduce_channels

        self.encoder_blocks = nn.ModuleList()
        for hidden_ch in hidden_channels:
            self.encoder_blocks.append(EncoderBlock(current_channels, hidden_ch, kernel_size))
            current_channels = hidden_ch

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_channels[-1],
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                bidirectional=True,
                dropout=dropout if lstm_num_layers > 1 else 0,
                batch_first=True,
            )
            current_channels = lstm_hidden_size * 2

        self.decoder_blocks = nn.ModuleList()
        decoder_channels = hidden_channels[::-1] + [32]
        for i in range(len(decoder_channels) - 1):
            self.decoder_blocks.append(DecoderBlock(current_channels, decoder_channels[i + 1], kernel_size))
            current_channels = decoder_channels[i + 1]

        self.final = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = x.transpose(1, 2)  # (N, L, C) -> (N, C, L)
        x = self.spatial_processing(x)
        skip_connections = []
        for encoder in self.encoder_blocks:
            x, features = encoder(x)
            skip_connections.append(features)
        return x, skip_connections

    def bottleneck(self, x):
        x = x.transpose(1, 2)  # (N, C, L) -> (N, L, C)
        x, _ = self.lstm(x)
        return x  # (N, L, C) — kept for cross-site attention

    def decode(self, x, skip_connections):
        x = x.transpose(1, 2)  # (N, L, C) -> (N, C, L)
        for decoder, skip in zip(self.decoder_blocks, skip_connections[::-1]):
            x = decoder(x, skip)
        x = self.final(x)
        return x.transpose(1, 2)  # (N, L, 1)

    def forward(self, x):
        x, skips = self.encode(x)
        x = self.bottleneck(x)
        return self.decode(x, skips)
