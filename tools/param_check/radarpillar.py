"""tools/param_check/radarpillar.py — 量 RadarPillar base 模型参数量并按模块拆解，对论文 0.27M 判定。

用法:
    PYTHONPATH=tools python tools/param_check/radarpillar.py \
        --cfg_file tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml

设计:
  - 量的是 requires_grad 的 trainable 参数。
  - 模型构建经 build_dataloader + build_network(dataset=...) 完成；
    build_dataloader 返回 (dataset, dataloader, sampler)，按方案取 index 0。
  - 按模块拆解（vfe / backbone_3d+map_to_bev / backbone_2d / dense_head）。
  - 直接参数 named-module 详单打印（但不与上面四组求和时双重计算）。
  - 容差: pass <=2% (264k-275k)、warn <=5%、fail >5% vs target 270000。
  - 含 OTHER 桶（任何不属于四组的 trainable 参数）。

共享的工具（count_params / per_module_breakdown / build_model_from_cfg /
verdict_pct）见同目录的 core.py；本脚本只保留 RadarPillar 特定的 group 归集
和 reporting。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path 前部（幂等）
# radarpillar.py 在 tools/param_check/，回退两级到仓库根
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from param_check.core import (  # noqa: E402  (param_check/ 已 PYTHONPATH=tools 暴露)
    build_model_from_cfg,
    count_trainable,
    per_module_breakdown,
    verdict_pct,
)


# 论文 Table V base (C=32) 全模型口径
TARGET_PARAMS = 270_000
PASS_PCT = 2.0
WARN_PCT = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description="参数量对账 (Task 3.5)")
    parser.add_argument("--cfg_file", type=str, required=True,
                        help="模型 yaml 路径（base：C=32 uniform）")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset, model, _cfg = build_model_from_cfg(
        args.cfg_file, training=False, batch_size=1, workers=0,
    )

    # ---- 总量 (trainable) ----
    total_trainable = count_trainable(model)
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
            n = sum(p.numel() for p in sub.parameters() if p.requires_grad)
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
                    type_group_counts[g] += sum(p.numel() for p in sub.parameters() if p.requires_grad)
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
                cnt += sum(p.numel() for p in sub.parameters() if p.requires_grad)
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
        n = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
        if n == 0:
            continue
        print(f"  {name:60s} {n:>10d}  {n/1e6:>9.5f} M")

    # ---- 容差判定 ----
    verdict = verdict_pct(total_trainable, TARGET_PARAMS, PASS_PCT, WARN_PCT)
    diff = total_trainable - TARGET_PARAMS
    abs_pct = abs(diff) / TARGET_PARAMS * 100.0
    signed_pct = diff / TARGET_PARAMS * 100.0

    print("\n" + "=" * 78)
    print(f"Diff vs target: {diff:+d} ({signed_pct:+.2f}% signed, {abs_pct:.2f}% abs)")
    print(f"Verdict: {verdict} "
          f"(PASS<={PASS_PCT}%, WARN<={WARN_PCT}%, FAIL>{WARN_PCT}%)")
    print("=" * 78)

    # 让 CI/外层脚本可程序化获取
    print(f"\n[VERDICT]: {verdict}")
    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())