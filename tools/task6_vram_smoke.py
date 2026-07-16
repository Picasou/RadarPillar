"""Task 6, Step 2 — VRAM smoke + safe batch_size extrapolation (audit M1).

Builds the model, runs ONE optimizer step at bs=1, records
torch.cuda.max_memory_allocated(), then linearly extrapolates the safe
batch_size for the 6.8GB (8GB*0.85) budget. Prints a recommendation.

Usage:
    python tools/task6_vram_smoke.py \
        --cfg_file tools/cfgs/model/vod_models/vod_radarnext_fpn.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from tools.utils.train_utils.optimization import build_optimizer


def measure_bs(bs, logger):
    common_utils.set_random_seed(666)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=bs, dist=False, workers=0, logger=logger,
        training=True, total_epochs=1,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda(); model.train()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    model_func = model_fn_decorator()
    batch = next(iter(train_loader))
    optimizer.zero_grad()
    loss, _, _ = model_func(model, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
    optimizer.step()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    del model, optimizer, train_loader, train_set, batch
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', type=str, required=True)
    args = ap.parse_args()
    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem

    GB = 1024 ** 3
    BUDGET_GB = 8 * 0.85  # 6.8 GB

    p1 = measure_bs(1, logger)
    logger.info('bs=1 peak = %.2f GiB (%.0f MiB)', p1 / GB, p1 / 1024**2)
    try:
        p2 = measure_bs(2, logger)
        logger.info('bs=2 peak = %.2f GiB (%.0f MiB)', p2 / GB, p2 / 1024**2)
        slope = (p2 - p1) / 1   # per-sample marginal cost
        intercept = p1 - slope  # fixed cost at bs=0
        safe_bs = int((BUDGET_GB * GB - intercept) / slope)
        logger.info('linear model: peak(bs) = %.2f GiB + %.2f GiB*bs', intercept/GB, slope/GB)
    except RuntimeError as e:
        logger.warning('bs=2 OOM: %s; using bs=1 only', e)
        slope = p1  # assume all-variable
        intercept = 0
        safe_bs = int((BUDGET_GB * GB) / slope)

    logger.info('budget = %.2f GiB (8GB x 0.85)', BUDGET_GB)
    logger.info('RECOMMENDED safe batch_size = %d', max(1, safe_bs))


if __name__ == '__main__':
    main()
