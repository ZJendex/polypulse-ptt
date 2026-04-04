"""Lightning module for single-site pulse detection training and evaluation."""

import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network import PulseDetectionNet
from .loss import PulseLoss
from .eval_metrics import PulseEval


class LitModel(pl.LightningModule):
    def __init__(self, config, debug=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.debug = debug
        self.model = PulseDetectionNet(**self.config.network)
        self.criterion = PulseLoss(**self.config.loss)
        self.evaluation = PulseEval(
            peak_min_distance=self.config.loss.min_peak_distance,
            site=self.config.data.position,
        )
        self.results = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}', value)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            self.log(f'val_{name}', value)
        _, count_error, all_distance_error, signed_all_distance_error = \
            self.evaluation.peak_error(y_hat, y, heights=[0.5])
        self.log('val_count_error', count_error[0], prog_bar=True)
        self.log('val_distance_error', np.median(all_distance_error[0]), prog_bar=True)
        self.log('val_signed_distance_error', np.median(signed_all_distance_error[0]), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, fnames = batch[0], batch[1], batch[2]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        for name, value in loss_components.items():
            self.log(f'test_{name}', value)

        heights, count_errors, all_distance_errors, signed_all_distance_errors = \
            self.evaluation.peak_error(y_hat, y)
        if self.thrs is None:
            self.thrs = heights
            self.distance_errs_at_thrs = [[] for _ in range(len(heights))]
            self.count_errs_at_thrs = [[] for _ in range(len(heights))]

        for idx, (height, count_error, distance_errors) in enumerate(
            zip(heights, count_errors, all_distance_errors)
        ):
            result_dict = {
                f'count_error_{height:.2f}': count_error,
                f'distance_error_{height:.2f}': np.median(distance_errors),
            }
            self.log_dict(result_dict, on_step=True, on_epoch=True)
            self.distance_errs_at_thrs[idx].extend(all_distance_errors[idx])
            self.count_errs_at_thrs[idx].append(count_error)

        if self.debug:
            unique_fnames, _ = np.unique(fnames, return_index=True)
            for fname in unique_fnames:
                mask = np.array(np.where(np.array(fnames) == fname)[0], dtype=int)
                _, ce, _, sde = self.evaluation.peak_error(y_hat[mask, :, :], y[mask, :, :], heights=[0.45])
                if fname not in self.debug_metrics:
                    self.debug_metrics[fname] = {'count_error': [], 'median_distance': [], 'median_abs_distance': []}
                self.debug_metrics[fname]['count_error'].append(ce[0])
                self.debug_metrics[fname]['median_distance'].extend(sde[0])
                self.debug_metrics[fname]['median_abs_distance'].extend(np.abs(sde[0]))

        return result_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        if self.config.scheduler.type == 'none':
            return {"optimizer": optimizer}
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config.scheduler.T_max, T_mult=1, eta_min=self.config.scheduler.min_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": 'epoch', "frequency": 1}}

    def on_test_start(self):
        self.thrs = None
        self.distance_errs_at_thrs = None
        self.count_errs_at_thrs = None
        self.debug_metrics = {}

    def on_test_epoch_end(self):
        for i in range(len(self.thrs)):
            self.distance_errs_at_thrs[i] = np.array(self.distance_errs_at_thrs[i])
            self.count_errs_at_thrs[i] = np.array(self.count_errs_at_thrs[i])

        if self.debug:
            for fname in self.debug_metrics:
                self.debug_metrics[fname]['count_error'] = np.mean(self.debug_metrics[fname]['count_error'])
                self.debug_metrics[fname]['median_distance'] = np.median(self.debug_metrics[fname]['median_distance'])
                self.debug_metrics[fname]['median_abs_distance'] = np.median(self.debug_metrics[fname]['median_abs_distance'])
        self.debug_metrics = [{'fname': f, **m} for f, m in self.debug_metrics.items()]

        self.results = {
            'thrs': np.array(self.thrs),
            'distance_errs_at_thrs': np.array(self.distance_errs_at_thrs, dtype=object),
            'count_errs_at_thrs': np.array(self.count_errs_at_thrs, dtype=object),
        }
