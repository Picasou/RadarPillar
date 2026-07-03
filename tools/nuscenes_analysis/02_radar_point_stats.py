"""雷达点云统计：点数 / 距离 / RCS / 速度。

对每个 RADAR_* 通道统计每帧点数，并汇总所有点的距离、RCS、|v| 分布。
输出图到 reports/figures/。

用法：
    python tools/nuscenes_analysis/02_radar_point_stats.py
    python tools/nuscenes_analysis/02_radar_point_stats.py --max_frames 1000
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # 无 GUI 也能画图
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


RADAR_CHANNELS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]
# 18 维 PCD 中需要用到的列
COL_X, COL_Y, COL_Z = 0, 1, 2
COL_RCS = 5
COL_VX, COL_VY = 6, 7


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataroot', default='data/nuscenes/v1.0-mini')
    ap.add_argument('--version', default='v1.0-mini')
    ap.add_argument('--max_frames', type=int, default=200,
                    help='统计帧数上限（mini 全跑也很快，trainval 抽 200 帧即可）')
    ap.add_argument('--out', default='reports/figures')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    n_per_chan = {ch: [] for ch in RADAR_CHANNELS}
    rng_all, rcs_all, vmag_all = [], [], []

    print(f'正在遍历前 {args.max_frames} 个关键帧 ...')
    for i, sample in enumerate(nusc.sample):
        if i >= args.max_frames:
            break
        for ch in RADAR_CHANNELS:
            sd_tok = sample['data'][ch]
            path, _, _ = nusc.get_sample_data(sd_tok)
            if not Path(path).exists():
                continue
            pc = RadarPointCloud.from_file(path)            # (18, N)
            n = pc.points.shape[1]
            n_per_chan[ch].append(n)

            xyz = pc.points[[COL_X, COL_Y, COL_Z], :]
            rng = np.linalg.norm(xyz, axis=0)
            rcs = pc.points[COL_RCS, :]
            vxy = pc.points[[COL_VX, COL_VY], :]
            vmag = np.linalg.norm(vxy, axis=0)

            rng_all.append(rng)
            rcs_all.append(rcs)
            vmag_all.append(vmag)

    rng_all = np.concatenate(rng_all)
    rcs_all = np.concatenate(rcs_all)
    vmag_all = np.concatenate(vmag_all)

    # 1) 每通道点数箱线图
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [n_per_chan[ch] for ch in RADAR_CHANNELS]
    ax.boxplot(
        data,
        tick_labels=[ch.replace('RADAR_', '') for ch in RADAR_CHANNELS],
    )
    ax.set_ylabel('points per frame')
    ax.set_title(f'Radar points per channel ({args.max_frames} frames sampled)')
    fig.tight_layout()
    fig.savefig(out_dir / 'radar_points_per_channel.png', dpi=120)
    plt.close(fig)

    # 2) 距离 / RCS / 速度 直方图
    for arr, name, xlabel in [
        (rng_all, 'range_m', 'range (m)'),
        (rcs_all, 'rcs_dbsm', 'RCS (dBsm)'),
        (vmag_all, 'vmag_ms', '|v| (m/s)'),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(arr, bins=80, color='steelblue', edgecolor='none')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.set_title(f'{name} distribution (total {len(arr):,} pts)')
        fig.tight_layout()
        fig.savefig(out_dir / f'radar_{name}.png', dpi=120)
        plt.close(fig)

    summary = {
        'n_frames_sampled': args.max_frames,
        'range_m': {
            'mean': float(rng_all.mean()),
            'p50': float(np.percentile(rng_all, 50)),
            'p95': float(np.percentile(rng_all, 95)),
            'max': float(rng_all.max()),
        },
        'rcs_dbsm': {
            'mean': float(rcs_all.mean()),
            'p50': float(np.percentile(rcs_all, 50)),
            'p95': float(np.percentile(rcs_all, 95)),
            'max': float(rcs_all.max()),
        },
        'vmag_ms': {
            'mean': float(vmag_all.mean()),
            'p50': float(np.percentile(vmag_all, 50)),
            'p95': float(np.percentile(vmag_all, 95)),
            'max': float(vmag_all.max()),
        },
        'points_per_chan_mean': {
            ch: float(np.mean(v)) if v else 0.0
            for ch, v in n_per_chan.items()
        },
    }
    print('---- Radar point statistics ----')
    print(json.dumps(summary, indent=2))

    (out_dir.parent / 'radar_point_stats.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f'\n[ok] figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
