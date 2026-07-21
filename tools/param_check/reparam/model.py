"""Task 5 — Reparam param-count verification (audit E).

Builds the RadarNeXt FPN-variant model from a YAML in TRAINING mode
(multi-branch RepDWC), calls ``reparameterize_model`` to fuse the
multi-branch blocks into the single-path inference graph, and reports
``sum(p.numel())`` for both. The INFERENCE-mode count is the figure the
RadarNeXt paper reports (target ≈ 0.899M, tolerance [0.854, 0.944]M).

Usage (PYTHONPATH=tools):
    python tools/param_check/reparam/model.py --cfg_file \
        tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml

The cfg is loaded the same way ``tools/train.py`` loads it
(``cfg_from_yaml_file`` over the module-level ``cfg``), and the model is
built via ``build_network`` mirroring ``train.py``'s call site. The
dataset is constructed only because ``build_network`` requires it for
``module_topology`` assembly (grid_size / point_cloud_range / num_rawpoint_features).

共享工具 (count_params / per_module_breakdown / build_model_from_cfg) 见
tools/param_check/core.py。
"""

import argparse
import sys
from pathlib import Path

import torch

# 仓库根加入 sys.path 前部（幂等）
# model.py 在 tools/param_check/reparam/，回退三级到仓库根
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pcdet.models.backbones_2d.mobileone_blocks import reparameterize_model
from pcdet.utils import common_utils

from param_check.core import (  # noqa: E402
    build_model_from_cfg,
    count_params,
    per_module_breakdown,
    verdict_pct,
)


def parse_args():
    parser = argparse.ArgumentParser(description='RadarNeXt reparam param-count check')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='model yaml (e.g. tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--breakdown', action='store_true',
                        help='print a per-module parameter breakdown for both modes')
    return parser.parse_args()


def main():
    args = parse_args()

    logger = common_utils.create_logger(log_file=None, rank=0)
    logger.info('Loading cfg: %s' % args.cfg_file)

    _dataset, model, _cfg = build_model_from_cfg(
        args.cfg_file, training=True, batch_size=args.batch_size, workers=args.workers,
        logger=logger,
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