#!/usr/bin/env python
"""从 train.log / train.log.gz 解析 loss 序列，画训练 loss 曲线。

用法:
    python tools/scripts/plot_loss.py [LOG_PATH] [OUT_PNG]

默认: LOG_PATH = output/debug_overfit_1batch/train.log
      OUT_PNG   = LOG_PATH 旁的 loss_curve.png
"""
import argparse
import gzip
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path: Path):
    """返回 (step_level_list, epoch_level_list)。
    step_level: [(total_it, loss), ...]
    epoch_level: [(epoch_idx, loss), ...]   # 同 epoch 多次出现，去重保留最后一次
    """
    loss_pat = re.compile(r"loss=([0-9.eE+\-]+)")
    totalit_pat = re.compile(r"total_it=(\d+)")
    # epochs:  N%|...| epoch/TOTAL [..., loss=X]
    # TOTAL 可以是任意正整数（80 / 100 / 30…）
    epoch_pat = re.compile(r"epochs:\s+(\d+)%\|.+?\|\s*(\d+)/(\d+)")

    steps = []
    epoch_seen = {}  # ep_idx -> loss (最后一次)
    opener = gzip.open if path.suffix == ".gz" else lambda p, *a, **kw: open(p, *a, **kw)
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_loss = loss_pat.search(line)
            if not m_loss:
                continue
            loss = float(m_loss.group(1))
            m_e = epoch_pat.search(line)
            if m_e:
                ep_curr = int(m_e.group(2))
                epoch_seen[ep_curr] = loss
            else:
                m_it = totalit_pat.search(line)
                if m_it:
                    steps.append((int(m_it.group(1)), loss))
    epoch_sorted = sorted(epoch_seen.items())  # [(1, lv), (2, lv)...]
    return steps, epoch_sorted


def plot_loss(steps, epoch_sorted, out: Path, title_suffix=""):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # (a) 全程 step-level loss + 平滑
    if steps:
        xs, ys = zip(*steps)
        axes[0].plot(xs, ys, lw=0.5, color="#1f77b4", alpha=0.5, label="train loss (per step)")
        if len(ys) > 30:
            window = max(20, len(ys) // 200)
            kernel = np.ones(window) / window
            ys_smooth = np.convolve(ys, kernel, mode="valid")
            axes[0].plot(xs[window - 1:], ys_smooth, lw=1.5, color="#d62728",
                         label=f"rolling mean (w={window})")
        axes[0].set_xlabel("training step (global)")
        axes[0].set_ylabel("total loss")
        axes[0].set_title(f"Training loss curve{title_suffix}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

    # (b) epoch-end loss
    if epoch_sorted:
        eps, eys = zip(*epoch_sorted)
        axes[1].plot(eps, eys, marker="o", lw=1.2, ms=3, color="#2ca02c")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("loss (sampled)")
        axes[1].set_title("Loss progression across epochs (tqdm snapshot)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(left=0)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    if epoch_sorted:
        eys = [v for _, v in epoch_sorted]
        print(f"saved: {out}")
        print(f"final loss: {eys[-1]:.4g}")
        print(f"min  loss:  {min(eys):.4g}")
    elif steps:
        ys = [v for _, v in steps]
        print(f"saved: {out}")
        print(f"final loss (step): {ys[-1]:.4g}")
        print(f"min  loss (step):  {min(ys):.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?",
                    default="output/debug_overfit_1batch/train.log",
                    help="train.log 或 .log.gz")
    ap.add_argument("out", nargs="?",
                    help="输出 PNG (默认: log 同目录的 loss_curve.png)")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        raise SystemExit(f"log not found: {log}")
    out = Path(args.out) if args.out else log.parent / "loss_curve.png"

    steps, epoch_sorted = parse_log(log)
    print(f"parsed: {len(steps)} step-level, {len(epoch_sorted)} epoch-unique")
    plot_loss(steps, epoch_sorted, out, title_suffix=f": {log.stem}")


if __name__ == "__main__":
    main()