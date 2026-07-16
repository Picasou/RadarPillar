"""Task 5 — Reparam param-count verification (audit E).

Builds the RadarNeXt FPN-variant model from a YAML in TRAINING mode
(multi-branch RepDWC), calls ``reparameterize_model`` to fuse the
multi-branch blocks into the single-path inference graph, and reports
``sum(p.numel())`` for both. The INFERENCE-mode count is the figure the
RadarNeXt paper reports (target ≈ 0.899M, tolerance [0.854, 0.944]M).

Usage (run from tools/):
    python reparam/reparam_model.py --cfg_file \
        cfgs/model/vod_models/vod_radarnext_fpn.yaml

The cfg is loaded the same way ``tools/train.py`` loads it
(``cfg_from_yaml_file`` over the module-level ``cfg``), and the model is
built via ``build_network`` mirroring ``train.py``'s call site. The
dataset is constructed only because ``build_network`` requires it for
``module_topology`` assembly (grid_size / point_cloud_range / num_rawpoint_features).
"""

import argparse
import os
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.models.backbones_2d.mobileone_blocks import reparameterize_model
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description='RadarNeXt reparam param-count check')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='model yaml (e.g. tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--breakdown', action='store_true',
                        help='print a per-module parameter breakdown for both modes')
    return parser.parse_args()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def per_module_breakdown(model, prefix=''):
    """Yield (name, numel) for every submodule that owns parameters directly."""
    rows = []
    own = sum(p.numel() for p in model.parameters(recurse=False))
    if own > 0:
        rows.append((prefix if prefix else '<root>', own))
    for name, child in model.named_children():
        child_prefix = f'{prefix}.{name}' if prefix else name
        rows.extend(per_module_breakdown(child, child_prefix))
    return rows


def main():
    args = parse_args()

    logger = common_utils.create_logger(log_file=None, rank=0)
    logger.info('Loading cfg: %s' % args.cfg_file)
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    log_config_to_file(cfg, logger=logger)

    # Build the dataset (training=True just to get a valid dataset object;
    # we never iterate it). build_network needs dataset.point_feature_encoder,
    # grid_size, point_cloud_range, voxel_size.
    logger.info('Building dataset (training split) for module_topology inputs...')
    train_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
        total_epochs=0,
    )

    logger.info('Building TRAINING-mode model (multi-branch RepDWC)...')
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=train_set,
    )
    # Keep on CPU — we are only counting parameters, no forward pass needed.
    model.train()

    train_params = count_params(model)
    logger.info('TRAINING-mode total params: %d (%.3fM)' %
                (train_params, train_params / 1e6))

    logger.info('Reparameterizing multi-branch -> inference single-path...')
    inference_model = reparameterize_model(model)
    inference_model.eval()
    inference_params = count_params(inference_model)
    logger.info('INFERENCE-mode total params: %d (%.3fM)' %
                (inference_params, inference_params / 1e6))

    target = 0.899e6
    lo, hi = 0.854e6, 0.944e6
    verdict = 'PASS' if lo <= inference_params <= hi else 'FAIL'
    logger.info('Target: %.3fM  tolerance: [%.3fM, %.3fM]  verdict: %s' %
                (target / 1e6, lo / 1e6, hi / 1e6, verdict))
    if train_params > 0:
        logger.info('Train/Inference ratio: %.2fx' % (train_params / inference_params))

    if args.breakdown:
        logger.info('-' * 60)
        logger.info('TRAINING-mode per-module breakdown:')
        for name, n in per_module_breakdown(model):
            logger.info('  %-60s %d' % (name, n))
        logger.info('-' * 60)
        logger.info('INFERENCE-mode per-module breakdown:')
        for name, n in per_module_breakdown(inference_model):
            logger.info('  %-60s %d' % (name, n))

    # Final machine-readable line for easy grepping.
    print('REPARAM_RESULT training=%d inference=%d target=%d verdict=%s'
          % (train_params, inference_params, int(target), verdict))


if __name__ == '__main__':
    main()
