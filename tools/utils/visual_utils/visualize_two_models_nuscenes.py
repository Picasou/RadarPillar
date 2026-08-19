#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nuScenes 双模型对比可视化 — visualize_nuscenes.py 同款统一格式。

同一帧出 [BEV+GT | BEV+model1 | BEV+model2] 三个 BEV 面板(+相机前排图),逐框对比两模型。
- 复用 visualize_nuscenes 的 infer_tokens(推理+LIDAR_TOP→ego 转换)/collect_all_radar_points/
  get_sample_gt_boxes_ego/_ego_to_screen,数据链路与单模型 viz 完全一致
- 框样式: 统一细实线空心,GT/两模型 pred 同类同色(CLASS_COLORS),区分只靠分面板+title 计数
- 点云: RCS viridis,三面板各画一份共用 colorbar

用法:
  python tools/utils/visual_utils/visualize_two_models_nuscenes.py \
      --nusc_dataroot /path/to/nuscenes \
      --cfg_file1 cfgs/.../m1.yaml --ckpt1 .../m1/best.pth --name1 baseline \
      --cfg_file2 cfgs/.../m2.yaml --ckpt2 .../m2/best.pth --name2 head2 \
      --split val --num 10 --out_dir output/cmp_nusc
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根

from visualize_nuscenes import (
    CLASS_COLORS, _ego_to_screen, collect_all_radar_points,
    get_sample_gt_boxes_ego, get_box_bev_corners, infer_tokens, pick_tokens,
    split_samples, _require_devkit,
)


def draw_pred_boxes_ego(ax, pboxes, R, score_thresh=0.1):
    """ego 系 pbox dict 列表 → 屏幕坐标画框(细实线空心,统一格式)。"""
    from matplotlib.patches import Polygon
    _, _, Box, _ = _require_devkit()
    from visualize_nuscenes import draw_bev_box
    n = 0
    for pbox in pboxes:
        if pbox.get('score', 1.0) < score_thresh:
            continue
        box = Box(pbox['translation'], pbox['size'],
                  _require_devkit()[3](pbox['rotation']))
        corners = get_box_bev_corners(box)
        sx, sy = _ego_to_screen(corners, R)
        color = CLASS_COLORS.get(pbox['det_cls'], '#95a5a6')
        draw_bev_box(ax, np.column_stack([sx, sy]), color,
                     linestyle='-', linewidth=1.2, alpha=0.9)
        n += 1
    return n


def setup_panel(ax, R):
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')
    ax.set_xlabel('Y [m] (left ← | right →)')
    ax.set_ylabel('X [m] (forward ↑)')
    ax.grid(True, alpha=0.3)
    ax.plot(0, 0, marker='^', color='white', markersize=10,
            markeredgecolor='black', zorder=5)


def plot_frame_compare(nusc, sample_token, dataroot, out_dir, name1, name2,
                       pb1, pb2, score_thresh, bev_range=55):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from visualize_nuscenes import draw_bev_box

    sample = nusc.get('sample', sample_token)
    radar_pts = collect_all_radar_points(nusc, sample, dataroot)
    gt_boxes = get_sample_gt_boxes_ego(nusc, sample)

    # 自适应范围: 点+GT+两模型 pred 联合外沿
    xs, ys = [], []
    if radar_pts.shape[1]:
        xs.append(radar_pts[0]); ys.append(radar_pts[1])
    for b in gt_boxes:
        c = get_box_bev_corners(b)
        xs.append(c[:, 0]); ys.append(c[:, 1])
    if xs:
        R = max(max(abs(np.concatenate(xs)).max(),
                    abs(np.concatenate(ys)).max()) + 5, bev_range)
    else:
        R = bev_range

    fig, panels = plt.subplots(1, 3, figsize=(21, 7.2),
                               gridspec_kw={'wspace': 0.22})

    # 点云: 三面板各一份, RCS viridis
    sc = None
    if radar_pts.shape[1]:
        sx, sy = _ego_to_screen(np.column_stack([radar_pts[0], radar_pts[1]]), R)
        for ax in panels:
            sc = ax.scatter(sx, sy, c=radar_pts[3], cmap='viridis',
                            s=4, alpha=0.6, zorder=1)
        cb = fig.colorbar(sc, ax=list(panels), fraction=0.03, pad=0.02)
        cb.set_label('RCS [dBsm]', fontsize=9)

    # GT 面板
    for box in gt_boxes:
        color = CLASS_COLORS.get(box.det_cls, '#95a5a6')
        corners = get_box_bev_corners(box)
        csx, csy = _ego_to_screen(corners, R)
        draw_bev_box(panels[0], np.column_stack([csx, csy]), color,
                     linestyle='-', linewidth=1.2, alpha=0.9)
    n1 = draw_pred_boxes_ego(panels[1], pb1 or [], R, score_thresh)
    n2 = draw_pred_boxes_ego(panels[2], pb2 or [], R, score_thresh)

    for ax in panels:
        setup_panel(ax, R)

    legend_classes = sorted(set(b.det_cls for b in gt_boxes)
                            | set(p['det_cls'] for p in (pb1 or []) if p['score'] >= score_thresh)
                            | set(p['det_cls'] for p in (pb2 or []) if p['score'] >= score_thresh))
    handles = [Line2D([0], [0], color=CLASS_COLORS.get(c, '#95a5a6'),
                      linewidth=2, label=c) for c in legend_classes]
    for ax, ttl in zip(panels, ['GT (%d)' % len(gt_boxes),
                                '%s (%d)' % (name1, n1), '%s (%d)' % (name2, n2)]):
        if handles:
            ax.legend(handles=handles, fontsize=7, loc='upper left', title='Class')
        ax.set_title(ttl, fontsize=10)

    fig.suptitle(f'Sample {sample_token[:12]}  |  GT {len(gt_boxes)}, '
                 f'{name1} {n1}, {name2} {n2}  (sc≥{score_thresh:.2f}, ±{R:.0f}m)',
                 fontsize=12, y=0.99)

    out = out_dir / ('%s_cmp_%s.png' % (sample['scene_token'][:6], sample_token[:12]))
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out, len(gt_boxes), n1, n2


def main():
    parser = argparse.ArgumentParser(description='nuScenes two-model comparison viz')
    parser.add_argument('--nusc_dataroot', required=True)
    parser.add_argument('--nusc_version', default='v1.0-mini')
    parser.add_argument('--cfg_file1', required=True)
    parser.add_argument('--ckpt1', required=True)
    parser.add_argument('--cfg_file2', required=True)
    parser.add_argument('--ckpt2', required=True)
    parser.add_argument('--name1', default='model1')
    parser.add_argument('--name2', default='model2')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--tokens', type=str, nargs='*', default=None)
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--bev_range', type=float, default=55)
    parser.add_argument('--out_dir', default='output/res_viz/two_models_nuscenes')
    args = parser.parse_args()

    NuScenes, _, _, _ = _require_devkit()
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_dataroot, verbose=False)

    samples = split_samples(nusc, args.split)
    tokens = args.tokens or pick_tokens(nusc, samples, args.num)

    print(f'=== two_models_nuscenes: split={args.split}, {len(tokens)} frames ===')
    pred1 = infer_tokens(nusc, args.cfg_file1, args.ckpt1, tokens)
    pred2 = infer_tokens(nusc, args.cfg_file2, args.ckpt2, tokens)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tok in tokens:
        out, n_gt, n1, n2 = plot_frame_compare(
            nusc, tok, args.nusc_dataroot, out_dir, args.name1, args.name2,
            pred1.get(tok), pred2.get(tok), args.score_thresh, args.bev_range)
        print(f'  {tok[:12]} -> {out}  (gt={n_gt} {args.name1}={n1} {args.name2}={n2})')
    print(f'=== done, output dir: {out_dir} ===')


if __name__ == '__main__':
    main()
