"""tools/count_params.py — 量 base 模型参数量并按模块拆解，对论文 0.27M 判定。

用法:
    PYTHONPATH=tools python tools/count_params.py \
        --cfg_file tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml

设计:
  - 量的是 requires_grad 的 trainable 参数。
  - 模型构建经 build_dataloader + build_network(dataset=...) 完成；
    build_dataloader 返回 (dataset, dataloader, sampler)，按方案取 index 0。
  - 按模块拆解（vfe / backbone_3d+map_to_bev / backbone_2d / dense_head）。
  - 直接参数 named-module 详单打印（但不与上面四组求和时双重计算）。
  - 容差: pass <=2% (264k-275k)、warn <=5%、fail >5% vs target 270000。
  - 含 OTHER 桶（任何不属于四组的 trainable 参数）。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# 在导入 pcdet 之前，将仓库根目录加入 sys.path 前部，
# 以保证加载本地 RadarPillar/pcdet/（含本仓库定制改动），而非 conda 系统中可能安装的 pcdet。
# 即便用户已经在 PYTHONPATH=tools 下调用，本段也是幂等的（仓库根已在前部）。
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


# 论文 Table V base (C=32) 全模型口径
TARGET_PARAMS = 270_000
PASS_PCT = 2.0
WARN_PCT = 5.0


def _trainable(mod):
    """Return trainable (requires_grad) parameter count for a module
    using `parameters(recurse=False)` semantics — i.e., only the direct
    parameters owned by this module, NOT children. Caller is responsible
    for deciding tree (root vs children)."""
    return sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)


def _descendant_trainable_count(root):
    """Total trainable params within an entire submodule subtree."""
    return sum(p.numel() for p in root.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="参数量对账 (Task 3.5)")
    parser.add_argument("--cfg_file", type=str, required=True,
                        help="模型 yaml 路径（base：C=32 uniform）")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = common_utils.create_logger(rank=0)
    # 屏蔽数据集/网络 verbose 噪声
    for h in logger.handlers:
        h.setLevel(logging.ERROR)
    logger.setLevel(logging.ERROR)

    cfg_from_yaml_file(args.cfg_file, cfg)

    # build_dataloader 返回 (dataset, dataloader, sampler) — 取 index 0
    dataset, _, _ = build_dataloader(
        cfg.DATA_CONFIG,
        cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
        total_epochs=1,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    # ---- 总量 (trainable) ----
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_m = total_trainable / 1e6

    print("=" * 78)
    print(f"TOTAL: {total_trainable} params ({total_m:.5f} M)")
    print(f"  target: {TARGET_PARAMS} ({TARGET_PARAMS/1e6:.5f} M) — paper RadarPillars base")
    print("=" * 78)

    # ---- 四组归集 (subtree sums, no double-counting) ----
    # Detector3DTemplate.module_topology (forward order):
    #   vfe, backbone_3d, map_to_bev_module, pfe, backbone_2d, dense_head,
    #   point_head, roi_head
    # 当 PFE/POINT_HEAD/ROI_HEAD 在 cfg 中未配置时，对应的 build_* 提前 return None，
    # 不进入 module_list。base 雷达 yaml 没有 PFE/POINT_HEAD/ROI_HEAD，因此 module_list
    # 仅含实际存在的 5 个子模块，按拓扑顺序追加：indices 实测 = 0..4 即
    #   PillarVFE / PillarAttention / PointPillarScatter / BaseBEVBackbone / AnchorHeadSingle。
    group_specs = {
        "vfe": (0,),                        # VFE
        "backbone_3d": (1, 2),              # BACKBONE_3D (PillarAttention) + MAP_TO_BEV
        "backbone_2d": (3,),                # BACKBONE_2D
        "dense_head": (4,),                 # DENSE_HEAD
    }

    modlist = model.module_list
    group_counts_fallback = None

    # 防御：长度不匹配时按类型自动归类并打印详情。
    expected_len = 5
    actual_len = len(modlist)
    if actual_len != expected_len:
        print(f"\n[NOTE] module_list length={actual_len}, expected={expected_len};")
        print(f"       PFE/POINT_HEAD/ROI_HEAD 已被 cfg 关闭或开启，导致拓扑相对期望有偏移。")
        for i, sub in enumerate(modlist):
            n = _descendant_trainable_count(sub)
            print(f"       [{i}] {type(sub).__name__:30s} {n:>8d} params")
        # 兜底按类型自动归类
        by_type = {
            "vfe": lambda t: 'PillarVFE' in t,
            "backbone_3d": lambda t: 'PillarAttention' in t or 'Scatter' in t,
            "backbone_2d": lambda t: 'BaseBEVBackbone' in t or 'SECOND' in t or 'FPN' in t,
            "dense_head": lambda t: 'AnchorHead' in t or 'Head' in t,
        }
        type_group_counts = {k: 0 for k in by_type}
        for sub in modlist:
            t = type(sub).__name__
            for g, pred in by_type.items():
                if pred(t):
                    type_group_counts[g] += _descendant_trainable_count(sub)
                    break
        group_counts_fallback = type_group_counts

    group_counts = {}
    if group_counts_fallback is not None:
        group_counts.update(group_counts_fallback)
    else:
        for name, idx_list in group_specs.items():
            cnt = 0
            for idx in idx_list:
                sub = modlist[idx]
                cnt += _descendant_trainable_count(sub)
            group_counts[name] = cnt

    # ---- OTHER 桶 ----
    accounted = sum(group_counts.values())
    other = total_trainable - accounted
    group_counts["OTHER"] = other

    print("\n[Group breakdown — 子树求和，不双重计算]")
    print("-" * 78)
    for name in ["vfe", "backbone_3d", "backbone_2d", "dense_head", "OTHER"]:
        c = group_counts[name]
        print(f"  {name:32s} {c:>10d}  {c/1e6:>9.5f} M")
    print("-" * 78)
    s = sum(v for k, v in group_counts.items() if k != "OTHER")
    print(f"  {'SUM-of-4 (no OTHER)':32s} {s:>10d}  {s/1e6:>9.5f} M")
    print(f"  {'TOTAL (cross-check)':32s} {total_trainable:>10d}  {total_m:>9.5f} M")
    assert s + other == total_trainable, (
        f"group+OTHER={s+other} != total={total_trainable}"
    )

    # ---- 命名模块详单 (direct params, no recursion) ----
    # 用于人工核对哪些 named-module 位于哪一组（避免误归类）。
    # 该详单**不**与组求和相加（每条只是 direct 参数，组求和用的是子树和）。
    print("\n[Named-module direct-param detail (no double-count in TOTAL)]")
    print("-" * 78)
    for name, mod in model.named_modules():
        n = _trainable(mod)
        if n == 0:
            continue
        print(f"  {name:60s} {n:>10d}  {n/1e6:>9.5f} M")

    # ---- 容差判定 ----
    diff = total_trainable - TARGET_PARAMS
    abs_pct = abs(diff) / TARGET_PARAMS * 100.0
    signed_pct = diff / TARGET_PARAMS * 100.0
    if abs_pct <= PASS_PCT:
        verdict = "PASS"
    elif abs_pct <= WARN_PCT:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    print("\n" + "=" * 78)
    print(f"Diff vs target: {diff:+d} ({signed_pct:+.2f}% signed, {abs_pct:.2f}% abs)")
    print(f"Verdict: {verdict} "
          f"(PASS<=2%, WARN<=5%, FAIL>5%)")
    print("=" * 78)

    # 让 CI/外层脚本可程序化获取
    print(f"\n[VERDICT]: {verdict}")
    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
