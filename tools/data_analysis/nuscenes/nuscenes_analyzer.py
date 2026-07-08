#!/usr/bin/env python3
"""nuScenes 数据分析统一入口

Usage:
    # 直接运行 (配置在脚本底部 CONFIG 区)
    python nuscenes_analyzer.py

    # 通过 yaml 配置文件
    python nuscenes_analyzer.py --config config.yaml

    # 通过命令行参数 (覆盖脚本配置)
    python nuscenes_analyzer.py --dataroot /path/to/data --source pkl --analysis basic_stats
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

# ══════════════════════════════════════════════════════════════
#  配置区域 (直接修改这里)
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # 数据源
    "dataroot": "/mnt/d/DATASET_PART",      # 数据根目录
    "pkl_root": None,                         # pkl 目录 (None=同 dataroot)
    "source": "pkl",                          # pkl / devkit / auto
    "version": "v1.0-mini",                   # devkit 模式用

    # 输出
    "output": "output/nuscenes_analysis",
    "max_samples": 0,                         # 0 = 全量
    "skip_plot": False,

    # 分析项目 (True = 启用)
    "analysis": {
        "basic_stats": True,
        "radar_distribution": True,
        "gt_stats": True,
        "bev_heatmap": True,
        "gt_pointcloud": True,    # 仅 pkl 模式有效
        "pts_in_gt_boxes": True,
    },
}


from analyzer_core import (
    detect_source,
    get_pkl_files,
    load_infos,
    get_nuscenes,
    analyze_basic,
    analyze_radar_distribution,
    analyze_gt_stats,
    analyze_bev_heatmap,
    analyze_gt_pointcloud,
    analyze_pts_in_gt_boxes,
    plot_all,
)

# 分析器注册表
ANALYZERS = {
    "basic_stats":        analyze_basic,
    "radar_distribution": analyze_radar_distribution,
    "gt_stats":           analyze_gt_stats,
    "bev_heatmap":        analyze_bev_heatmap,
    "gt_pointcloud":      analyze_gt_pointcloud,
    "pts_in_gt_boxes":    analyze_pts_in_gt_boxes,
}


def parse_args():
    parser = argparse.ArgumentParser(description="nuScenes 数据分析工具")
    parser.add_argument("--config", type=str, default=None,
                        help="yaml 配置文件路径")
    # 数据源
    parser.add_argument("--dataroot", type=str, default=None,
                        help="数据根目录")
    parser.add_argument("--pkl_root", type=str, default=None,
                        help="pkl 文件目录 (若与 dataroot 不同)")
    parser.add_argument("--source", type=str, choices=["pkl", "devkit", "auto"], default="auto",
                        help="数据源: pkl/devkit/auto (自动检测)")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes 版本 (devkit 模式用)")
    # 分析项目
    parser.add_argument("--analysis", nargs="+", default=None,
                        help="要运行的分析项目: basic_stats, radar_distribution, gt_stats, bev_heatmap, gt_pointcloud, pts_in_gt_boxes")
    # 输出
    parser.add_argument("--out", type=str, default="output/nuscenes_analysis",
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="限制样本数 (0=全量)")
    parser.add_argument("--skip_plot", action="store_true",
                        help="跳过可视化")
    return parser.parse_args()


def load_config(config_path):
    """加载 yaml 配置。"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def merge_config_and_args(cfg, args):
    """合并配置文件和命令行参数, 命令行参数优先。"""
    # 数据源
    if args.dataroot:
        cfg.dataroot = args.dataroot
    if args.pkl_root:
        cfg.pkl_root = args.pkl_root
    if args.source != "auto":
        cfg.source = args.source
    if args.version:
        cfg.version = args.version

    # 分析项目
    if args.analysis:
        cfg.analysis = {k: True for k in args.analysis}

    # 输出
    if args.out:
        cfg.output = args.out
    if args.max_samples:
        cfg.max_samples = args.max_samples
    if args.skip_plot:
        cfg.skip_plot = True

    return cfg


def get_analysis_list(cfg):
    """获取要运行的分析项目列表。"""
    default_projects = list(ANALYZERS.keys())
    if not hasattr(cfg, "analysis") or cfg.analysis is None:
        return default_projects

    analysis_cfg = cfg.analysis
    if isinstance(analysis_cfg, dict):
        return [k for k, v in analysis_cfg.items() if v]
    elif isinstance(analysis_cfg, list):
        return analysis_cfg
    return default_projects


def run_analysis(source_type, source_obj, analysis_list, cfg):
    """运行分析项目 — 纯调度，数据加载在各 analyze_* 内部完成。"""
    results = {"source": source_type, "version": cfg.get("version", "v1.0-mini")}
    total = len(analysis_list)

    for idx, name in enumerate(analysis_list, 1):
        if name not in ANALYZERS:
            print(f"  [{idx}/{total}] {name}: UNKNOWN — 跳过")
            continue
        print(f"  [{idx}/{total}] Running {name}...")
        result = ANALYZERS[name](source_type, source_obj, cfg)
        results[name] = result

        # 简要打印结果
        _print_summary(name, result)

    return results


def _print_summary(name, result):
    """打印分析结果的简要摘要。"""
    if result is None:
        print(f"        [SKIP]")
        return

    summary_map = {
        "basic_stats": lambda r: f"samples={r.get('n_sample', 'N/A')}, annos={r.get('n_anno', 'N/A')}",
        "radar_distribution": lambda r: f"n_frames={r.get('n_frames', 'N/A')}",
        "gt_stats": lambda r: f"n_gt_boxes={r.get('n_gt_boxes', 'N/A')}",
        "bev_heatmap": lambda r: f"n_centers={r.get('n', 'N/A')}",
        "gt_pointcloud": lambda r: f"n_classes={len(r)}" if r else "[SKIP] 仅支持 pkl 模式",
        "pts_in_gt_boxes": lambda r: f"n_classes={len(r)}" if r else "[SKIP]",
    }

    printer = summary_map.get(name)
    if printer:
        print(f"        {printer(result)}")


def main():
    args = parse_args()

    # ── 加载配置 ─────────────────────────────────────────────────
    if args.config:
        cfg = load_config(args.config)
    else:
        # 默认使用脚本顶部的 CONFIG 配置
        cfg = EasyDict(CONFIG.copy())

    cfg = merge_config_and_args(cfg, args)

    # 必要参数检查
    if not cfg.get("dataroot"):
        print("ERROR: 必须指定 --dataroot 或在配置文件中设置 dataroot")
        sys.exit(1)

    print(f"=" * 60)
    print(f"nuScenes 数据分析")
    print(f"  dataroot: {cfg.dataroot}")
    print(f"  source:   {cfg.source}")
    print(f"=" * 60)

    # ── 检测数据源 ───────────────────────────────────────────────
    if cfg.source == "auto":
        source_type = detect_source(cfg.dataroot, cfg.get("pkl_root"))
        print(f"  自动检测数据源: {source_type}")
    else:
        source_type = cfg.source
        # 验证
        if source_type == "pkl":
            get_pkl_files(cfg.dataroot, cfg.get("pkl_root"))
        else:
            get_nuscenes(cfg.dataroot, cfg.version)

    # ── 加载数据对象 ─────────────────────────────────────────────
    if source_type == "pkl":
        pkl_files = get_pkl_files(cfg.dataroot, cfg.get("pkl_root"))
        print(f"  加载 pkl: {len(pkl_files)} files")
        source_obj = load_infos(pkl_files)
        print(f"  样本数: {len(source_obj)}")
    else:
        print(f"  加载 nuScenes {cfg.version}...")
        source_obj = get_nuscenes(cfg.dataroot, cfg.version)
        print(f"  scenes={len(source_obj.scene)}, samples={len(source_obj.sample)}")

    # ── 确定分析项目 ─────────────────────────────────────────────
    analysis_list = get_analysis_list(cfg)
    print(f"  分析项目: {', '.join(analysis_list)}")
    print(f"=" * 60)

    # ── 运行分析 ─────────────────────────────────────────────────
    results = run_analysis(source_type, source_obj, analysis_list, cfg)

    # ── 保存结果 ─────────────────────────────────────────────────
    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {json_path}")

    # ── 可视化 ───────────────────────────────────────────────────
    if not cfg.get("skip_plot", False):
        print("\nGenerating plots...")
        plot_all(results, out_dir)

    print("\nDONE.")


if __name__ == "__main__":
    main()
