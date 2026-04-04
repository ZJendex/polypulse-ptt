"""Multi-site pulse detection with optional cross-site temporal attention.

Output modes:
  - default:      (N, num_sites, L, 1) per-site pulse probabilities
  - direct_ptt:   (N, num_pairs) scalar PTT regression values
"""

import torch
import torch.nn as nn
from .network import PulseDetectionNet


class CrossSiteTemporalAttention(nn.Module):
    """Two-stage MHA: cross-site attention at each timestep, then temporal attention per site."""

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.site_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, site_features):
        # site_features: list of (N, L, C) tensors
        stacked = torch.stack(site_features, dim=1)  # (N, S, L, C)
        N, S, L, C = stacked.shape

        # Cross-site attention at each timestep: (N*L, S, C)
        reshaped = stacked.transpose(1, 2).reshape(N * L, S, C)
        site_attn, _ = self.site_attention(reshaped, reshaped, reshaped)
        site_attn = self.norm1(site_attn + reshaped)

        # Temporal attention per site: (N*S, L, C)
        temporal_input = site_attn.reshape(N, L, S, C).transpose(1, 2).reshape(N * S, L, C)
        temporal_attn, _ = self.temporal_attention(temporal_input, temporal_input, temporal_input)
        temporal_attn = self.norm2(temporal_attn + temporal_input)

        output = temporal_attn.reshape(N, S, L, C)
        return [output[:, i] for i in range(S)]


class PTTRegressionHead(nn.Module):
    """MLP: concatenated site features -> scalar PTT value."""

    def __init__(self, in_features, hidden_features=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x):
        return self.fc(x)


class MultiSitePulseDetectionNet(nn.Module):
    def __init__(self, site_configs, enable_fusion=True, direct_ptt=False, pairs=None):
        super().__init__()
        self.num_sites = len(site_configs)
        self.site_networks = nn.ModuleList([PulseDetectionNet(**c) for c in site_configs])
        self.site_in_channels = [c['in_channels'] for c in site_configs]

        # Bidirectional LSTM doubles the hidden size
        lstm_hidden_size = site_configs[0]['lstm_hidden_size'] * 2

        self.enable_fusion = enable_fusion
        self.direct_ptt = direct_ptt

        if self.enable_fusion:
            self.cross_site_attention = CrossSiteTemporalAttention(hidden_size=lstm_hidden_size)

        if self.direct_ptt:
            self.all_pairs = pairs or [
                (i, j) for i in range(self.num_sites) for j in range(i + 1, self.num_sites)
            ]
            self.ptt_regression_heads = nn.ModuleDict({
                f'{p[0]}_{p[1]}': PTTRegressionHead(2 * lstm_hidden_size, lstm_hidden_size)
                for p in self.all_pairs
            })

    def forward(self, site_inputs):
        # site_inputs: (N, num_sites, spatial, temporal)
        encoded, skips = [], []
        for i, net in enumerate(self.site_networks):
            feat, skip = net.encode(site_inputs[:, i, :, :self.site_in_channels[i]])
            feat = net.bottleneck(feat)
            encoded.append(feat)
            skips.append(skip)

        fused = self.cross_site_attention(encoded) if self.enable_fusion else encoded

        if self.direct_ptt:
            pooled = [f.mean(dim=1) for f in fused]  # list of (N, C)
            return torch.cat([
                self.ptt_regression_heads[f'{p[0]}_{p[1]}'](
                    torch.cat([pooled[p[0]], pooled[p[1]]], dim=1)
                ) for p in self.all_pairs
            ], dim=1)  # (N, num_pairs)

        return torch.stack([
            net.decode(fused[i], skips[i]) for i, net in enumerate(self.site_networks)
        ], dim=1)  # (N, num_sites, L, 1)

    def load_pretrained(self, paths):
        """Load per-site weights from individual Lightning checkpoints."""
        for network, path in zip(self.site_networks, paths):
            sd = torch.load(path)['state_dict']
            network.load_state_dict({k.replace('model.', ''): v for k, v in sd.items()})

    def freeze_site(self, site_idx):
        for p in self.site_networks[site_idx].parameters():
            p.requires_grad = False

    def unfreeze_site(self, site_idx):
        for p in self.site_networks[site_idx].parameters():
            p.requires_grad = True

    def freeze_all_sites(self):
        for i in range(self.num_sites):
            self.freeze_site(i)
