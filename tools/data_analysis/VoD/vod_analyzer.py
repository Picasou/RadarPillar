#!/usr/bin/env python3
"""VoD 数据分析统一入口

使用:
    # 单进程分析 (analyze -> merge -> report 一步完成)
    python vod_analyzer.py --dataroot /path/to/radar_5frames --split trainval --out output

    # 分片分析 (跨分片合并, 适合大样本并行)
    python vod_analyzer.py --dataroot /path/to/radar_5frames --split train --shard 0 --nshards 4 --mode analyze --out output
    ...
    python vod_analyzer.py --dataroot /path/to/radar_5frames --split train --mode report --partials output/train/partial_*.json --out output

    # 限样本快速测试
    python vod_analyzer.py --dataroot /path/to/radar_5frames --split train --limit 20 --out output/smoke
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

# ══════════════════════════════════════════════════════════════
#  默认配置
# ══════════════════════════════════════════════════════════════
CONFIG = {
    "dataroot": "/mnt/d/DATASET/VoD/view_of_delft_PUBLIC/view_of_delft_PUBLIC/radar_5frames",
    "split": "trainval",
    "output": "output",
    "shard": 0,
    "nshards": 1,
    "limit": 0,
    "mode": "all",     # all / analyze / report
    "partials": None,  # 手动指定 partial json glob 模式
    "raw_labels": True,  # report 阶段是否扫原始 label_2/*.txt
}

from analyzer_core import (
    load_infos,
    split_root,
    analyze_shard,
    merge_partials,
    compute_report,
    plot_all,
    write_markdown,
    write_csvs,
    scan_raw_labels,
)


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="VoD radar 数据分析工具")
    p.add_argument("--config", type=str, default=None, help="yaml 配置文件路径")
    p.add_argument("--dataroot", type=str, default=None, help="数据根目录 (含 vod_infos_*.pkl 与 training/)")
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test", "trainval"])
    p.add_argument("--out", dest="output", type=str, default=None, help="输出目录")
    p.add_argument("--shard", type=int, default=None, help="当前分片索引 (0-based)")
    p.add_argument("--nshards", type=int, default=None, help="总分片数")
    p.add_argument("--limit", type=int, default=None, help="限制样本数 (0=全量)")
    p.add_argument("--mode", type=str, default=None, choices=["all", "analyze", "report"])
    p.add_argument("--partials", type=str, default=None, help="report 模式下的 partial json glob 模式")
    p.add_argument("--no_raw_labels", action="store_true", help="report 阶段跳过原始 label_2 扫描")
    return p.parse_args()


def merge(cfg, args):
    for k in ("dataroot", "split", "output", "shard", "nshards", "limit", "mode", "partials"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.no_raw_labels:
        cfg["raw_labels"] = False
    return cfg


def _to_json_safe(obj):
    """递归把 numpy 标量/数组转成 Python 原生类型, 便于 json.dump。"""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    return obj


def cmd_analyze(cfg):
    """单分片分析: 读 pkl, 取本片, 输出 partial json。"""
    print("=" * 60)
    print(f"VoD analyze  mode=analyze  split={cfg.split}  shard={cfg.shard}/{cfg.nshards}  limit={cfg.limit}")
    print("=" * 60)

    infos = load_infos(cfg.dataroot, cfg.split)
    print(f"  加载 infos: {len(infos)} samples")

    root = split_root(cfg.dataroot, cfg.split)
    part = analyze_shard(infos, root, shard=cfg.shard, nshards=cfg.nshards, limit=cfg.limit,
                         desc=f"shard{cfg.shard}/{cfg.nshards}")
    part["_meta"] = {"split": cfg.split, "shard": cfg.shard, "nshards": cfg.nshards, "limit": cfg.limit}

    out_dir = Path(cfg.output) / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"partial_{cfg.shard}.json"
    fp.write_text(json.dumps(_to_json_safe(part), ensure_ascii=False))
    print(f"  Saved partial -> {fp}  (n_frames={part['n_frames']})")
    return fp


def cmd_report(cfg):
    """合并 partials -> summary + CSVs + plots + report.md。"""
    print("=" * 60)
    print(f"VoD analyze  mode=report  split={cfg.split}")
    print("=" * 60)

    out_dir = Path(cfg.output) / cfg.split
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.partials:
        paths = sorted(glob.glob(cfg.partials))
    else:
        paths = sorted(glob.glob(str(out_dir / "partial_*.json")))
    if not paths:
        print(f"ERROR: no partials found in {out_dir}")
        sys.exit(1)
    print(f"  Merging {len(paths)} partials:")
    for p in paths:
        print(f"    - {p}")

    parts = [json.loads(Path(p).read_text()) for p in paths]
    merged = merge_partials(parts)
    n_box = sum(len(merged["per_class"][c].get("live", [])) for c in merged["per_class"])
    print(f"  Merged: {merged['n_frames']} frames, {n_box} boxes")

    # 14 类原始分布
    raw_counts, raw_files = {}, 0
    if cfg.get("raw_labels", True):
        label_dir = split_root(cfg.dataroot, cfg.split) / "label_2"
        if label_dir.exists():
            print(f"  Scanning raw labels: {label_dir}")
            raw_files, raw_counts = scan_raw_labels(label_dir)
            print(f"    -> {raw_files} files, {len(raw_counts)} classes")
        else:
            print(f"  [warn] no label_2 dir: {label_dir}")

    report = compute_report(merged, cfg.split, raw_label_counts=raw_counts, raw_label_files=raw_files)

    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  Saved summary -> {out_dir / 'summary.json'}")

    write_csvs(report, out_dir)
    plot_all(report, merged, out_dir)
    write_markdown(report, out_dir)
    print("\nDONE.")


def cmd_all(cfg):
    cfg.shard, cfg.nshards = 0, 1
    cmd_analyze(cfg)
    cfg.partials = str(Path(cfg.output) / cfg.split / "partial_0.json")
    cmd_report(cfg)


def main():
    args = parse_args()
    if args.config:
        cfg = EasyDict(yaml.safe_load(open(args.config)))
    else:
        cfg = EasyDict(CONFIG.copy())
    cfg = merge(cfg, args)

    if not cfg.dataroot:
        print("ERROR: 必须指定 --dataroot")
        sys.exit(1)

    mode = cfg.mode
    if mode == "analyze":
        cmd_analyze(cfg)
    elif mode == "report":
        cmd_report(cfg)
    else:
        cmd_all(cfg)


if __name__ == "__main__":
    main()
