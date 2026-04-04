"""Loss functions for pulse detection (BCE with Gaussian targets) and direct PTT regression (MSE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseLoss(nn.Module):
    def __init__(self,
                 seq_len: int,
                 sigma: float = 2.0,
                 min_peak_distance: int = 0.5 * 500,
                 max_peak_distance: int = 1.2 * 500,
                 distance_weight: float = 0.1,
                 count_weight: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.sigma = sigma
        self.min_peak_distance = min_peak_distance
        self.max_peak_distance = max_peak_distance
        self.distance_weight = distance_weight
        self.count_weight = count_weight

    def generate_gaussian_target(self, peak_locations, seq_len):
        """Place a Gaussian bump at each ground-truth peak location."""
        device = peak_locations.device
        time_idx = torch.arange(seq_len, device=device).float()
        gaussian_target = torch.zeros_like(peak_locations, dtype=torch.float32)

        peak_indices = torch.where(peak_locations > 0)
        if peak_indices[0].shape[0] > 0:
            for b, t in zip(peak_indices[0], peak_indices[1]):
                gaussian = torch.exp(-0.5 * ((time_idx - t.float()) / self.sigma) ** 2)
                gaussian_target[b, :, 0] += gaussian.squeeze()

        return torch.clamp(gaussian_target, 0, 1)

    def forward(self, pred, target, name=None):
        gaussian_target = self.generate_gaussian_target(target, self.seq_len)
        bce_loss = F.binary_cross_entropy(pred, gaussian_target)
        return bce_loss, {'bce_loss': bce_loss.item()}


class MultiSitePulseLoss(nn.Module):
    def __init__(self, config_list, weights=None, names=['head', 'heart', 'wrist']):
        super().__init__()
        self.losses = nn.ModuleList([PulseLoss(**config) for config in config_list])
        if weights is None:
            weights = [1.0] * len(config_list)
        self.weights = torch.tensor(weights)
        self.names = names

    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[0].device)
        loss_components = {}
        for i, (loss, name) in enumerate(zip(self.losses, self.names)):
            loss_value, components = loss(preds[:, i, :, :], targets[:, i, :, :])
            total_loss += self.weights[i] * loss_value
            for key, value in components.items():
                loss_components[f'{name}_{key}'] = value
        return total_loss, loss_components


class DirectPTTRegressionLoss(nn.Module):
    def __init__(self, pairs):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.pairs = pairs

    def forward(self, preds, targets):
        mse_loss = self.criterion(preds, targets)
        pair_losses = {}
        for i, pair in enumerate(self.pairs):
            pair_losses[f'mse_loss_{pair[0]}_{pair[1]}'] = F.mse_loss(preds[:, i], targets[:, i]).item()
        return mse_loss, pair_losses
