"""Lightning module for multi-site joint pulse detection and PTT evaluation."""

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import os
import logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .network_joint import MultiSitePulseDetectionNet
from .loss import MultiSitePulseLoss
from .eval_metrics import PulseEval


class LitModel_joint(pl.LightningModule):
    def __init__(self, config_list, training_config, checkpoint_paths=None,
                 debug=False, enable_fusion=True, disable_print=False):
        super().__init__()
        self.save_hyperparameters()
        self.config_list = config_list
        self.training_config = training_config
        self.sites_names = [config.data.position for config in config_list]
        self.num_sites = len(config_list)
        self.debug = debug
        self.disable_print = disable_print

        if disable_print:
            logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
            self.trainer_kwargs = {"enable_progress_bar": False}
        else:
            self.trainer_kwargs = {}

        self.network_configs = [config.network for config in config_list]
        self.loss_configs = [config.loss for config in config_list]

        self.model = MultiSitePulseDetectionNet(self.network_configs, enable_fusion=enable_fusion)
        self.criterion = MultiSitePulseLoss(self.loss_configs, names=self.sites_names)
        self.evaluation = PulseEval(peak_min_distance=self.loss_configs[0].min_peak_distance)

        if checkpoint_paths is not None:
            self.model.load_pretrained(checkpoint_paths)
            self.model.freeze_all_sites()

        self.results = None
        self.username = None
        self.ptt_queries = [
            (1, 2, -95, -35),
            (1, 3, -40, 10),
            (0, 3, 0, 40),
        ]
        self.height_thrs = [0.35, 0.55, 0.25, 0.5]  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        if not self.disable_print:
            self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=False, on_epoch=True)
            for name, value in loss_components.items():
                self.log(f'train_{name}', value)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)
        if not self.disable_print:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            for name, value in loss_components.items():
                self.log(f'val_{name}', value)
            for i in range(self.num_sites):
                _, count_error, distance_errors, _ = \
                    self.evaluation.peak_error(y_hat[:, i, :, :], y[:, i, :, :], heights=[0.5])
                self.log(f'val_count_error_{self.sites_names[i]}', count_error[0], prog_bar=True)
                self.log(f'val_distance_error_{self.sites_names[i]}', np.median(distance_errors[0]), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, names = batch[0], batch[1], batch[2]
        y_hat = self(x)
        loss, loss_components = self.criterion(y_hat, y)

        if np.unique(names).shape[0] == 1:
            bname = names[0]
            if not self.disable_print:
                print(bname)
            if self.username is None:
                self.username = bname
                os.makedirs('../results', exist_ok=True)

        if not self.disable_print:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            for name, value in loss_components.items():
                self.log(f'test_{name}', value)

        for i in range(self.num_sites):
            heights, count_errors, all_distance_errors, _ = \
                self.evaluation.peak_error(y_hat[:, i, :, :], y[:, i, :, :])
            if len(self.distance_errs_at_thrs[self.sites_names[i]]) == 0:
                self.thrs = heights
                self.distance_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
                self.count_errs_at_thrs[self.sites_names[i]] = [[] for _ in range(len(heights))]
            for idx, (height, count_error, dist_errs) in enumerate(zip(heights, count_errors, all_distance_errors)):
                self.distance_errs_at_thrs[self.sites_names[i]][idx].extend(dist_errs)
                self.count_errs_at_thrs[self.sites_names[i]][idx].append(count_error)
                if not self.disable_print:
                    self.log_dict({
                        f'{self.sites_names[i]}_count_error_{height:.2f}': count_error,
                        f'{self.sites_names[i]}_distance_error_{height:.2f}': np.median(dist_errs),
                    }, on_step=True, on_epoch=True)

        ptt_metrics, ptt_samples = self.evaluation.ptt_error(
            y_hat, y, ptt_queries=self.ptt_queries, height_thrs=self.height_thrs)
        for i in range(len(self.ptt_queries)):
            self.ptt_detect_rates[i]['gt_ptt'].append(ptt_metrics[i]['gt_ptt_rate'])
            self.ptt_detect_rates[i]['pred_ptt'].append(ptt_metrics[i]['pred_ptt_rate'])
            # *2 converts from samples to ms at 500 Hz
            ptt_gt = abs(ptt_samples[i]['gt_ptt'] * 2)
            ptt_pred = abs(ptt_samples[i]['pred_ptt'] * 2)
            self.median_ptt_batch[i]['gt_ptt'].append(np.median(ptt_gt))
            self.median_ptt_batch[i]['pred_ptt'].append(np.median(ptt_pred))
            self.ptt_samples[i]['gt_ptt'].append(ptt_gt)
            self.ptt_samples[i]['pred_ptt'].append(ptt_pred)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.training.learning_rate,
            weight_decay=self.training_config.training.weight_decay,
        )
        if self.training_config.scheduler.type == 'none':
            return {"optimizer": optimizer}
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.training_config.scheduler.T_max, T_mult=1,
            eta_min=self.training_config.scheduler.min_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": 'epoch', "frequency": 1}}

    def on_test_start(self):
        self.thrs = None
        self.distance_errs_at_thrs = {site: [] for site in self.sites_names}
        self.count_errs_at_thrs = {site: [] for site in self.sites_names}
        self.ptt_samples = [{'gt_ptt': [], 'pred_ptt': []} for _ in self.ptt_queries]
        self.ptt_detect_rates = [{'gt_ptt': [], 'pred_ptt': []} for _ in self.ptt_queries]
        self.median_ptt_batch = [{'gt_ptt': [], 'pred_ptt': []} for _ in self.ptt_queries]

    def on_test_epoch_end(self):
        thrs = np.array(self.thrs)
        self.results = {'thrs': thrs}
        for si, site in enumerate(self.sites_names):
            for i in range(len(self.thrs)):
                self.distance_errs_at_thrs[site][i] = np.array(self.distance_errs_at_thrs[site][i])
                self.count_errs_at_thrs[site][i] = np.array(self.count_errs_at_thrs[site][i])
            closest_idx = np.argmin(np.abs(self.thrs - self.height_thrs[si]))
            self.results[f'{site}_distance_errs_at_thrs'] = self.distance_errs_at_thrs[site][closest_idx]
            self.results[f'{site}_count_errs_at_thrs'] = self.count_errs_at_thrs[site][closest_idx]

        data = {}
        for i in range(len(self.ptt_queries)):
            s1 = self.sites_names[self.ptt_queries[i][0]]
            s2 = self.sites_names[self.ptt_queries[i][1]]
            col = f'ptt_{s1}_{s2}'
            data[f'{col}_gt'] = self.median_ptt_batch[i]['gt_ptt']
            data[f'{col}_pred'] = self.median_ptt_batch[i]['pred_ptt']

        bp_path = f'../results/gt_bp/{self.username}.csv'
        if os.path.exists(bp_path):
            bp_gt = pd.read_csv(bp_path, sep='\t')
            data['sys'] = bp_gt['sys']
            data['dia'] = bp_gt['dia']

        df = pd.DataFrame(data)
        df.to_csv(f'../results/{self.username}_ptt_results.csv', index=False)
        print("Results Saved:")
        print(df.iloc[0].to_dict())

    def get_trainer_kwargs(self):
        return self.trainer_kwargs
