#!/usr/bin/env python3
"""VoD eval 后处理: 结构化结果 + 损失曲线 + 多帧可视化

用法:
    # 三件套
    python tools/visualize_eval.py --eval_dir <...>/eval/epoch_100/val/val_eval \
        --dataroot /mnt/d/DATASET/VoD/.../radar_5frames \
        --train_log_dir output/cfgs/model/vod_models/vod_radarpillar/<EXTRA_TAG>

    # 只画 loss
    python tools/visualize_eval.py ... --loss_only

    # 只画帧
    python tools/visualize_eval.py ... --frames_only --n_samples 12 --score_thresh 0.2

    # 重新生成 results.json (基于 log_eval_*.txt)
    python tools/visualize_eval.py ... --results_only
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# 让同目录下的 tools/ 模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.visual_utils.visualize_vod_eval import (
    compose_one_frame, iter_sample_ids_uniform, load_frame_assets, lookup_predictions_for_frame,
)
from utils.visual_utils.visualize_loss import parse_log, visualize_loss as plot_loss_from_log


# ══════════════════════════════════════════════════════════════
#  results.json / results.csv
# ══════════════════════════════════════════════════════════════
def parse_results_from_text(result_str):
    """解析 test.py logger 打出的 result_str。"""
    # 形如:
    # Car AP@0.50, 0.25, 0.25:
    #   bbox AP:0.7000, 0.6500, 0.6000
    #   bev  AP:...
    #   3d   AP:...
    #   aos  AP:...
    # Car AP_R40@0.50, 0.25, 0.25:
    #   ...
    per_class = {}
    cur_cls, cur_tag = None, None
    for line in result_str.splitlines():
        m = re.match(r"(\w+)\s+AP(_R40)?@([\d\., ]+):", line)
        if m:
            cur_cls, r40 = m.group(1), bool(m.group(2))
            cur_tag = "AP_R40" if r40 else "AP"
            per_class.setdefault(cur_cls, {}).setdefault(cur_tag, {})
            continue
        m2 = re.match(r"\s*(bbox|bev|3d|aos)\s+AP:([\d\., -]+)", line)
        if m2 and cur_cls and cur_tag:
            vals = [float(x) for x in m2.group(2).replace(" ", "").split(",") if x]
            per_class[cur_cls][cur_tag][m2.group(1)] = {
                "easy": vals[0] if len(vals) > 0 else 0,
                "moderate": vals[1] if len(vals) > 1 else 0,
                "hard": vals[2] if len(vals) > 2 else 0,
            }
    return per_class


def write_results(eval_dir, output_dir):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 优先用 test.py 落盘的 results.json
    j = eval_dir / "results.json"
    per_class = {}
    ret_dict = {}
    summary = ""
    if j.exists():
        d = json.loads(j.read_text())
        per_class = d.get("per_class", {})
        ret_dict = d.get("ret_dict", {})
        summary = d.get("summary_str", "")
        if not per_class and summary:
            per_class = parse_results_from_text(summary)
    else:
        # fallback: 解析最新 log_eval_*.txt
        logs = sorted(eval_dir.glob("log_eval_*.txt"))
        if not logs:
            print(f"  [warn] no results.json or log_eval_*.txt in {eval_dir}")
            return None
        text = logs[-1].read_text(errors="ignore")
        per_class = parse_results_from_text(text)
        summary = text

    (output_dir / "results.json").write_text(
        json.dumps({"per_class": per_class, "ret_dict": ret_dict, "summary_str": summary},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    # CSV
    lines = ["class,task,difficulty,AP,AP_R40"]
    for cls, d in per_class.items():
        for task in ("bbox", "bev", "3d", "aos"):
            for diff in ("easy", "moderate", "hard"):
                ap = d.get("AP", {}).get(task, {}).get(diff)
                ap_r40 = d.get("AP_R40", {}).get(task, {}).get(diff)
                if ap is None and ap_r40 is None:
                    continue
                lines.append(f"{cls},{task},{diff},{ap if ap is not None else ''},{ap_r40 if ap_r40 is not None else ''}")
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  results -> {output_dir / 'results.json'}")
    print(f"  results -> {output_dir / 'results.csv'}")
    return per_class


# ══════════════════════════════════════════════════════════════
#  loss 曲线
# ══════════════════════════════════════════════════════════════
def plot_tb_loss_curves(train_log_dir, output_dir):
    """从 TB events 画 rpn_loss / cls / loc / dir / total / lr 多子图。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [skip] tensorboard not installed, skip TB loss curves")
        return

    tb_dir = Path(train_log_dir) / "tensorboard"
    if not tb_dir.exists():
        print(f"  [skip] no tensorboard dir: {tb_dir}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    wanted = ["train/rpn_loss", "train/rpn_loss_cls", "train/rpn_loss_loc",
              "train/rpn_loss_dir", "train/loss", "meta_data/learning_rate"]
    available = [t for t in wanted if t in tags]
    if not available:
        print(f"  [warn] no target tags in TB. available: {tags}")
        return

    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, tag in enumerate(available):
        events = ea.Scalars(tag)
        xs = [e.step for e in events]
        ys = [e.value for e in events]
        axes[i].plot(xs, ys, lw=0.8, color="#1f77b4")
        axes[i].set_title(tag)
        axes[i].set_xlabel("step"); axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"TensorBoard scalars ({tb_dir.parent.name})")
    plt.tight_layout()
    out = output_dir / "tb_loss_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  TB loss -> {out}")


def plot_log_loss_curve(train_log_dir, output_dir):
    """复用 visualize_loss.parse_log + visualize_loss 画 train.log 总 loss。"""
    log_dir = Path(train_log_dir) / "logs"
    if not log_dir.exists():
        print(f"  [skip] no logs dir: {log_dir}")
        return
    logs = sorted(log_dir.glob("train_*.log"))
    if not logs:
        print(f"  [skip] no train_*.log in {log_dir}")
        return
    log = logs[-1]
    steps, epoch_sorted = parse_log(log)
    if not steps and not epoch_sorted:
        print(f"  [skip] no loss lines parsed from {log}")
        return
    out = output_dir / "loss_curve.png"
    plot_loss_from_log(steps, epoch_sorted, out, title_suffix=f": {log.stem}")
    print(f"  log loss -> {out}")


# ══════════════════════════════════════════════════════════════
#  多帧可视化
# ══════════════════════════════════════════════════════════════
def visualize_n_frames(eval_dir, dataroot, output_dir, n_samples, score_thresh, seed=42):
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    infos_pkl = Path(dataroot) / "vod_infos_val.pkl"
    if not infos_pkl.exists():
        print(f"  [error] missing {infos_pkl}")
        return

    frame_ids = iter_sample_ids_uniform(infos_pkl, n_samples, seed=seed)
    result_pkl = eval_dir / "result.pkl"

    n_ok = 0
    for fid in frame_ids:
        try:
            pts, img, calib, gt_lidar, gt_names = load_frame_assets(dataroot, "val", fid)
            pred_lidar, pred_names, pred_scores = lookup_predictions_for_frame(result_pkl, fid) \
                if result_pkl.exists() else (np.zeros((0, 7)), [], np.array([]))
            out = output_dir / f"frame_{fid}.png"
            compose_one_frame(fid, pts, img, calib, gt_lidar, gt_names,
                              pred_lidar, pred_names, pred_scores,
                              out, score_thresh=score_thresh)
            print(f"    frame {fid} -> {out}")
            n_ok += 1
        except Exception as e:
            print(f"    [skip] frame {fid}: {e}")
    print(f"  done: {n_ok}/{len(frame_ids)} frames")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True, help="output/.../eval/epoch_<N>/val/<eval_tag>/")
    p.add_argument("--dataroot", required=True, help="VoD radar_5frames 根目录")
    p.add_argument("--train_log_dir", default=None, help="含 logs/ 和 tensorboard/ 的目录 (默认 = eval_dir 的 ../..)")
    p.add_argument("--output_dir", default=None, help="可视化输出 (默认 = eval_dir/vis/)")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--score_thresh", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="跑 results + loss + frames (默认)")
    g.add_argument("--results_only", action="store_true")
    g.add_argument("--loss_only", action="store_true")
    g.add_argument("--frames_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "vis"
    train_log_dir = Path(args.train_log_dir) if args.train_log_dir else eval_dir.parent.parent.parent

    only = sum(bool(x) for x in (args.results_only, args.loss_only, args.frames_only))
    if only == 0:
        args.all = True
    run_results = args.all or args.results_only
    run_loss = args.all or args.loss_only
    run_frames = args.all or args.frames_only

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=" * 60)
    print(f"VoD eval postprocess")
    print(f"  eval_dir:      {eval_dir}")
    print(f"  train_log_dir: {train_log_dir}")
    print(f"  output_dir:    {output_dir}")
    print(f"=" * 60)

    if run_results:
        print("\n[1/3] results.json / results.csv")
        write_results(eval_dir, output_dir)

    if run_loss:
        print("\n[2/3] loss curves")
        plot_log_loss_curve(train_log_dir, output_dir)
        plot_tb_loss_curves(train_log_dir, output_dir)

    if run_frames:
        print(f"\n[3/3] frame visualization (n={args.n_samples}, score_thresh={args.score_thresh})")
        visualize_n_frames(eval_dir, args.dataroot, output_dir,
                           args.n_samples, args.score_thresh, seed=args.seed)

    print("\nDONE.")


if __name__ == "__main__":
    main()
