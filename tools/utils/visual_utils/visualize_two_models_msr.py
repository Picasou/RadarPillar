#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MSR 双模型对比可视化 — visualize_msr.py 同款统一格式。

同一帧出 [GT | model1 | model2] 三个 BEV 面板(可选相机图在左),逐框对比两模型预测。
- 直接吃两个 ckpt 实时推理(走 viz_common.build_viz_net,与单模型 viz 同链路)
- 框样式: 三面板同款细实线空心,GT/两模型 pred 同类同色,区分只靠分面板+title 计数
- 点云: doppler_gnd diverging 蓝↔红(与 visualize_msr.py 同款),三面板各画一份共用 colorbar

用法:
  python tools/utils/visual_utils/visualize_two_models_msr.py \
      --cfg_file1 experiments/MC_DATASET/YAML/msr.yaml \
      --ckpt1 output/.../model1/best.pth --name1 baseline \
      --cfg_file2 experiments/MC_DATASET/YAML/msr_head2.yaml \
      --ckpt2 output/.../model2/best.pth --name2 head2 \
      --split val --num 10 --out_dir output/cmp_msr
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'scripts' / 'data'))

from visualize_msr import (CLASS_COLOR, CLASS_LABEL, DIV_BLUE, DIV_GRAY, DIV_RED,
                           GRID, INK, INK2, MUTED, pick_frames)


def plot_frame_compare(ds1, ds2, sid, split, out_dir, name1, name2,
                       pred1, pred2, color_by='doppler_gnd'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from viz_common import draw_box_bev

    # 两数据集对象可能 cfg 不同但同帧 raw 数据一致;点云/GT 取 ds1
    pts = ds1.get_radar(sid)
    cols = {n: i for i, n in enumerate(ds1.radar_feature_order)}
    boxes, names = ds1.get_label(sid)

    # ---- 布局: [相机 | GT | model1 | model2] ----
    img_file = ds1.root_split_path / 'IMAGES' / ('%s.png' % sid)
    if img_file.exists():
        from skimage import io
        fig, axes = plt.subplots(1, 4, figsize=(26, 6.6),
                                 width_ratios=[0.9, 1, 1, 1],
                                 gridspec_kw={'wspace': 0.28})
        ax_img, panels = axes[0], list(axes[1:])
        ax_img.imshow(io.imread(str(img_file)))
        ax_img.set_title('Camera', fontsize=10, color=INK2, pad=8)
        ax_img.axis('off')
    else:
        fig, panels = plt.subplots(1, 3, figsize=(20, 6.6), gridspec_kw={'wspace': 0.25})

    # ---- 点云: 三面板同款 diverging ----
    if pts.shape[0]:
        if color_by == 'doppler_gnd':
            azi = pts[:, cols['ang_rad']]
            c = (pts[:, cols['dop_x_gnd']] * np.cos(azi)
                 + pts[:, cols['dop_y_gnd']] * np.sin(azi))
        else:
            c = pts[:, cols[color_by]]
        vmax = np.percentile(np.abs(c), 99) or 1.0
        cmap = LinearSegmentedColormap.from_list('msr_div', [DIV_BLUE, DIV_GRAY, DIV_RED])
        norm = Normalize(vmin=-vmax, vmax=vmax)
        for ax in panels:
            ax.scatter(pts[:, 1], pts[:, 0], c=c, cmap=cmap, norm=norm,
                       s=2, linewidths=0, alpha=0.9, zorder=2)
        sc = panels[-1].collections[-1]
        cb = fig.colorbar(sc, ax=list(panels), fraction=0.03, pad=0.02)
        cb.set_label(color_by, fontsize=9, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)

    # ---- 框: GT 面板画 GT, model 面板各画自己 pred(同款细实线) ----
    cls_names = ['1', '2', '4', '5']

    def draw_boxes(ax, items):
        for b, n in items:
            n = str(n)
            color = CLASS_COLOR.get(n, '#eda100')
            draw_box_bev(ax, b, color, linestyle='-', linewidth=1.2,
                         zorder=4, swap_xy=True)

    draw_boxes(panels[0], list(zip(boxes, names)))
    n1 = n2 = 0
    if pred1 is not None:
        lb2name = [cls_names[int(lb) - 1] if 0 < int(lb) <= 4 else str(lb)
                   for lb in pred1.get('pred_labels', [])]
        draw_boxes(panels[1], list(zip(pred1.get('pred_boxes', []), lb2name)))
        n1 = len(pred1.get('pred_boxes', []))
    if pred2 is not None:
        lb2name = [cls_names[int(lb) - 1] if 0 < int(lb) <= 4 else str(lb)
                   for lb in pred2.get('pred_labels', [])]
        draw_boxes(panels[2], list(zip(pred2.get('pred_boxes', []), lb2name)))
        n2 = len(pred2.get('pred_boxes', []))

    # ---- 轴样式: 与 visualize_msr.py 统一 (x 前/y 左, 联合外沿+3m, 等比) ----
    if pts.shape[0] or boxes.shape[0]:
        xs = np.concatenate([pts[:, 0], boxes[:, 0]]) if pts.shape[0] else boxes[:, 0]
        ys = np.concatenate([pts[:, 1], boxes[:, 1]]) if pts.shape[0] else boxes[:, 1]
        m = 3.0
        xlo, xhi = min(xs.min(), 0) - m, xs.max() + m
        ylo, yhi = min(ys.min(), 0) - m, ys.max() + m
    else:
        xlo, xhi, ylo, yhi = -10, 10, -10, 10
    for ax in panels:
        ax.set_xlim(yhi, ylo)  # 显示横轴 = y (顺减 → +y 朝左)
        ax.set_ylim(xlo, xhi)
        ax.set_aspect('equal')
        ax.set_xlabel('y (m)', fontsize=9, color=INK2)
        ax.set_ylabel('x (m)', fontsize=9, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.scatter([0], [0], marker='o', s=5, color=INK, zorder=5)

    # 图例 + title 计数 (GT(n) / name1(n) / name2(n))
    class_handles = [Line2D([0], [0], color=CLASS_COLOR[k], linewidth=2,
                            label=CLASS_LABEL.get(k, k)) for k in ['1', '2', '4', '5']]
    for ax, ttl in zip(panels, ['GT (%d)' % boxes.shape[0],
                                '%s (%d)' % (name1, n1), '%s (%d)' % (name2, n2)]):
        ax.legend(handles=class_handles, loc='upper right', fontsize=8,
                  framealpha=0.9, edgecolor=GRID, labelcolor=INK2,
                  title='Class', title_fontsize=9)
        ax.set_title(ttl, fontsize=10, color=INK2, pad=8)

    fig.suptitle('MSR %s  idx=%s  |  GT %d, %s %d, %s %d'
                 % (split, sid, boxes.shape[0], name1, n1, name2, n2),
                 fontsize=12, color=INK, y=0.99)

    out = out_dir / ('%s_cmp_%s.png' % (split, sid))
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out, boxes.shape[0], n1, n2


def main():
    parser = argparse.ArgumentParser(description='MSR two-model comparison viz')
    parser.add_argument('--cfg_file1', required=True, help='模型1 cfg')
    parser.add_argument('--ckpt1', required=True, help='模型1 ckpt(best.pth)')
    parser.add_argument('--cfg_file2', required=True, help='模型2 cfg')
    parser.add_argument('--ckpt2', required=True, help='模型2 ckpt(best.pth)')
    parser.add_argument('--name1', default='model1')
    parser.add_argument('--name2', default='model2')
    parser.add_argument('--split', default='val', choices=['training', 'val', 'both'])
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--indices', type=str, nargs='*', default=None)
    parser.add_argument('--out_dir', default='output/res_viz/two_models_msr')
    parser.add_argument('--color_by', default='doppler_gnd',
                        choices=['doppler_gnd', 'doppler_mps', 'rcs', 'z', 'range_m'])
    args = parser.parse_args()

    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.msr.msr_dataset import MsrDataset
    from viz_common import build_viz_net, infer_sample_ids

    def build(cfg_file, ckpt, split):
        mcfg = cfg_from_yaml_file(cfg_file, EasyDict())
        mcfg.DATA_CONFIG.DATA_SPLIT.test = split
        if split == 'training':
            mcfg.DATA_CONFIG.INFO_PATH.test = list(mcfg.DATA_CONFIG.INFO_PATH.train)
        ds = MsrDataset(dataset_cfg=mcfg.DATA_CONFIG, class_names=list(mcfg.CLASS_NAMES),
                        training=False, root_path=None)
        net = build_viz_net(cfg_file, ckpt)[0] if ckpt else None
        return ds, net

    splits = ['training', 'val'] if args.split == 'both' else [args.split]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        ds1, net1 = build(args.cfg_file1, args.ckpt1, split)
        ds2, net2 = build(args.cfg_file2, args.ckpt2, split)
        ids = args.indices or pick_frames(ds1, ds1.sample_id_list, args.num, split=split)
        print('=== two_models_msr: split=%s, %d frames ===' % (split, len(ids)))
        for sid in ids:
            p1 = infer_sample_ids(net1, ds1, [sid]).get(sid) if net1 is not None else None
            p2 = infer_sample_ids(net2, ds2, [sid]).get(sid) if net2 is not None else None
            out, n_gt, n1, n2 = plot_frame_compare(
                ds1, ds2, sid, split, out_dir, args.name1, args.name2, p1, p2, args.color_by)
            print('  %s -> %s  (gt=%d %s=%d %s=%d)' % (sid, out, n_gt, args.name1, n1,
                                                       args.name2, n2))
    print('=== done, output dir: %s ===' % out_dir)


if __name__ == '__main__':
    main()
