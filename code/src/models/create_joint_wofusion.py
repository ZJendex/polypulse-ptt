"""Assemble a joint checkpoint from individually trained per-site checkpoints (without fusion)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from models.lightning_model_joint import LitModel_joint
from config.config_utils import load_config
import torch
import argparse


def create_joint_model(configs, checkpoint_paths, output_path, disable_print=False):
    config_default = load_config('src/config', 'joint')
    pl.seed_everything(configs[0].training.seed, workers=True)

    model = LitModel_joint(configs, training_config=config_default,
                           checkpoint_paths=checkpoint_paths, enable_fusion=False,
                           disable_print=disable_print)

    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': {'config_list': configs, 'training_config': config_default},
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': pl.__version__,
        'callbacks': {},
        'optimizer_states': [],
        'lr_schedulers': [],
        'NativeMixedPrecisionPlugin': None,
        'version': None,
    }
    torch.save(checkpoint, output_path)
    if not disable_print:
        print(f'Saved joint model to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Assemble joint checkpoint from per-site checkpoints.')
    parser.add_argument('--head-checkpoint', type=str, help='Head site checkpoint path')
    parser.add_argument('--heart-checkpoint', type=str, help='Heart site checkpoint path')
    parser.add_argument('--wrist-checkpoint', type=str, help='Wrist site checkpoint path')
    parser.add_argument('--neck-checkpoint', type=str, help='Neck site checkpoint path')
    parser.add_argument('--head-model-config', type=str, default='head')
    parser.add_argument('--heart-model-config', type=str, default='heart')
    parser.add_argument('--wrist-model-config', type=str, default='wrist')
    parser.add_argument('--neck-model-config', type=str, default='neck')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint path')
    parser.add_argument('--disable-print', action='store_true', default=False)
    args = parser.parse_args()

    checkpoints = [args.head_checkpoint, args.heart_checkpoint, args.wrist_checkpoint, args.neck_checkpoint]
    configs = [
        load_config('src/config', env=args.head_model_config),
        load_config('src/config', env=args.heart_model_config),
        load_config('src/config', env=args.wrist_model_config),
        load_config('src/config', env=args.neck_model_config),
    ]
    create_joint_model(configs, checkpoints, args.output, disable_print=args.disable_print)


if __name__ == '__main__':
    main()
