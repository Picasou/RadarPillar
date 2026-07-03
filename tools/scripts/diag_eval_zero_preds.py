#!/usr/bin/env python
"""验证 overfit 1-batch 是否真的不泛化。

用法:
    python tools/scripts/diag_eval_zero_preds.py

预期结果（overfit 单 batch 的本质）：
    train[0]  → pred_after_nms > 0  （见过的样本可预测）
    val[0]    → pred_after_nms = 0  （没见过的样本预测不出来）
    → 结论: 0-preds 是 overfit 单 batch 的天然结果，不是代码 bug

如果不是上面这个模式，说明真有问题：本脚本会把差异打印出来。
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/dministrator1/RadarPillar")
sys.path.insert(0, str(REPO / "tools"))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

CKPT = REPO / "output/cfgs/nuscenes_models/radarpillar_nuscenes/debug_overfit_1batch/ckpt/checkpoint_epoch_100.pth"
CFG = REPO / "tools/cfgs/nuscenes_models/radarpillar_nuscenes.yaml"


def run_one(model, sample, name):
    model.eval()
    # Inspect voxels BEFORE forward to nail the bug
    if 'voxels' in sample:
        v = sample['voxels']
        print(f"  [{name}] voxels.shape={tuple(v.shape)}, dtype={v.dtype}, type={type(v)}")
    gt = sample.get('gt_boxes', None)
    if gt is not None:
        print(f"  [{name}] gt_boxes.shape={tuple(gt.shape)}  (gt count per sample)")
        for i in range(min(2, gt.shape[0])):
            valid = gt[i][gt[i, :, 7] >= 0] if gt.shape[-1] >= 8 else gt[i]
            print(f"           sample {i}: {valid.shape[0]} valid GTs, "
                  f"first cls labels: {valid[:5, 7].cpu().tolist() if valid.shape[0]>0 else 'none'}")
    with torch.no_grad():
        pred_dicts, _ = model(sample)
    # 打印 pred 分数分布
    for i, pd in enumerate(pred_dicts):
        if 'pred_scores' in pd:
            s = pd['pred_scores']
            n_raw = s.shape[0]
            if n_raw > 0:
                print(f"  [{name}] sample {i}: {n_raw} boxes after NMS, "
                      f"score stats: max={s.max().item():.3f}, "
                      f"mean={s.mean().item():.3f}, "
                      f">0.1={(s > 0.1).sum().item()}, "
                      f">0.05={(s > 0.05).sum().item()}")
            else:
                print(f"  [{name}] sample {i}: 0 boxes after NMS")
    # pred_dicts: list[dict], one per batch element
    counts = []
    sample_sizes = []
    for pd in pred_dicts:
        if 'pred_boxes' in pd:
            n = pd['pred_boxes'].shape[0]
            scores = pd.get('pred_scores', torch.tensor([]))
            sample_sizes.append(int(scores.shape[0]) if hasattr(scores, 'shape') else len(scores))
        else:
            n = pd.get('pred_boxes_bev', torch.tensor([])).shape[0] if 'pred_boxes_bev' in pd else -1
        counts.append(int(n) if isinstance(n, int) else int(n.item()) if hasattr(n, 'item') else int(n))
    print(f"  [{name}] pred_keys = {list(pred_dicts[0].keys())[:8]}...")
    print(f"  [{name}] pred counts (after NMS): {counts}, scores: {sample_sizes}")
    return counts


def main():
    cfg_from_yaml_file(str(CFG), cfg)
    logger = common_utils.create_logger()
    # build train loader
    cfg.DATA_CONFIG.DATA_SPLIT['test'] = cfg.DATA_CONFIG.DATA_SPLIT['train']
    ds, dl, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=True,
    )
    # build val loader (reuse ds but training=False)
    ds_v, dl_v, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False,
    )

    # build model + load ckpt (build_network needs num_class + dataset)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    sd = torch.load(str(CKPT), map_location='cpu', weights_only=False)
    state_key = 'model_state_dict' if 'model_state_dict' in sd else 'model_state'
    model.load_state_dict(sd[state_key])
    model.cuda()

    print(f"ckpt = {CKPT.name}")
    print(f"train_ds size={len(ds)}, val_ds size={len(ds_v)}")

    # 取 train[0] — 用与训练一致的 load_data_to_gpu（numpy → torch.cuda）
    from pcdet.models import load_data_to_gpu
    print("\n=== Train[0] (被训练过 100 次) ===")
    batch_train = next(iter(dl))
    load_data_to_gpu(batch_train)
    train_counts = run_one(model, batch_train, "train[0]")

    # 取 val[0]
    print("\n=== Val[0] (未见过) ===")
    batch_val = next(iter(dl_v))
    load_data_to_gpu(batch_val)
    val_counts = run_one(model, batch_val, "val[0]")

    # 对比
    print("\n=== Verdict ===")
    if train_counts[0] > 0 and val_counts[0] == 0:
        print(f"  train>0 AND val=0 → 符合 overfit 单 batch 的天然结果。")
        print(f"  不是代码 bug。把 batch_size 拉回正常做真正训练即可。")
    elif train_counts[0] > 0 and val_counts[0] > 0:
        print(f"  train>0 AND val>0 → 模型有泛化。train.log 的 0 是评估间隔的随机状态。")
    elif train_counts[0] == 0:
        print(f"  train=0 AND val=0 → 模型本就没学到东西（score_thresh 过滤）。")
    else:
        print(f"  train=0 AND val>0 → 反过来，奇怪。需进一步排查。")


if __name__ == '__main__':
    main()
