"""GT 类别 / 距离 / 速度 / 尺寸分布（10 类 detection 标签）。

用法：
    python tools/nuscenes_analysis/03_gt_distribution.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--out', default='reports/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cls_dist: dict[str, int] = defaultdict(int)
    range_by_cls: dict[str, list[float]] = defaultdict(list)
    speed_by_cls: dict[str, list[float]] = defaultdict(list)
    size_by_cls: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for sample in nusc.sample:
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            raw_name = ann['category_name']  # devkit 1.0.5 直接给
            name = map_name_from_general_to_detection.get(raw_name, 'ignore')
            if name == 'ignore':
                continue
            cls_dist[name] += 1

            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            range_by_cls[name].append(float(np.linalg.norm(box.center[:2])))
            vel = nusc.box_velocity(ann_tok)
            if vel is not None and not np.isnan(vel).any():
                speed_by_cls[name].append(float(np.linalg.norm(vel[:2])))
            size_by_cls[name].append(tuple(box.wlh))  # (w, l, h)

    # 1) 类别直方图
    names = list(cls_dist.keys())
    counts = [cls_dist[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color='seagreen')
    ax.set_ylabel('count')
    ax.set_title('GT class distribution')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_class_dist.png', dpi=120)
    plt.close(fig)

    # 2) 每个类别的距离直方图（叠在一起）
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(range_by_cls.keys()):
        arr = np.array(range_by_cls[n])
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=50, alpha=0.45, label=f'{n}({len(arr)})')
    ax.set_xlabel('range (m)')
    ax.set_ylabel('count')
    ax.set_title('GT range by class')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_range_by_class.png', dpi=120)
    plt.close(fig)

    # 3) 每个类别的速度分布
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in sorted(speed_by_cls.keys()):
        arr = np.array(speed_by_cls[n])
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=50, alpha=0.45, label=f'{n}({len(arr)})')
    ax.set_xlabel('|v| (m/s)')
    ax.set_ylabel('count')
    ax.set_title('GT speed by class')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'gt_speed_by_class.png', dpi=120)
    plt.close(fig)

    summary = {
        'class_count': dict(sorted(cls_dist.items(), key=lambda x: -x[1])),
        'range_m_mean': {
            n: float(np.mean(range_by_cls[n]))
            for n in names if range_by_cls[n]
        },
        'speed_mps_mean': {
            n: float(np.mean(speed_by_cls[n]))
            for n in names if speed_by_cls[n]
        },
        'size_lwh_mean': {
            n: np.mean(size_by_cls[n], axis=0).tolist()  # wlh 顺序
            for n in names if size_by_cls[n]
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    (out_dir.parent / 'gt_distribution.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f'\n[ok] figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
