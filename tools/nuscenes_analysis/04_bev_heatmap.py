"""BEV 占用热力图：把所有帧的 GT 中心画在 BEV 栅格里。

用法：
    python tools/nuscenes_analysis/04_bev_heatmap.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes

from pcdet.datasets.nuscenes.nuscenes_utils import map_name_from_general_to_detection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--range', type=float, default=50.0,
                    help='BEV 半边长 (米)')
    ap.add_argument('--out', default='reports/figures/bev_heatmap.png')
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    centers = []
    for sample in nusc.sample:
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            raw = ann['category_name']  # devkit 1.0.5 直接给
            if map_name_from_general_to_detection.get(raw, 'ignore') == 'ignore':
                continue
            centers.append(ann['translation'][:2])  # x, y
    centers = np.array(centers)

    R = args.range
    H, _, _ = np.histogram2d(
        centers[:, 0], centers[:, 1],
        bins=[100, 100],
        range=[[-R, R], [-R, R]],
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        H.T, origin='lower', extent=[-R, R, -R, R],
        cmap='hot', interpolation='bilinear',
    )
    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('y (m, left)')
    ax.set_title(f'GT center BEV heatmap ({len(centers)} boxes, range=±{R}m)')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'[ok] heatmap saved -> {out}')


if __name__ == '__main__':
    main()
