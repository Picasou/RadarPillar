#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KITTI 系(KITTI / VoD / Astyx)可视化脚本 — visualize_msr.py 同款格式。

三面板: [相机 | BEV+GT | BEV+pred],GT 与 pred 分面板独立展示,便于对比。
- BEV 朝向: x 前(屏幕上) / y 左(屏幕左),车规惯例,与 visualize_msr.py 一致
- 点着色:   used_feature_list 中速度/多普勒列 diverging 对称色标,rcs/intensity 顺序色
- 框样式:   GT 与 pred 同款细实线空心(区分只靠分面板+title 计数),与 visualize_msr.py 一致
- 类色:     避开点云蓝/红色域(黄/品红/草绿/青绿族),框与点云色相分离
- 选帧:     MSR 式智能选帧(分段覆盖+类多样性+签名去重),train/val 各抽 --num 帧
- 输出:     默认 output/res_viz/<cfg名>/;图名 <split>_bev_<idx>.png

用法:
  # 纯数据模式(抽查)
  python tools/utils/visual_utils/visualize_kitti.py \
      --cfg_file experiments/YAML/a1.yaml --split both --num 10
  # 训练后叠加预测(full_chain viz step)
  python tools/utils/visual_utils/visualize_kitti.py \
      --cfg_file experiments/YAML/a1.yaml --ckpt output/.../best.pth --split both --num 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根

# categorical 固定槽位(按 CLASS_NAMES 出现顺序分配;同类 GT/pred 同色)
# 黄/品红/草绿/青绿族,避开点云 diverging(蓝↔红)色域 — 与 visualize_msr.py 同族
PALETTE = ['#f1c40f', '#d63ee0', '#7cb342', '#00b3a4', '#e67e22',
           '#9b59b6', '#f39c12', '#8a7d3b', '#c8b8db', '#7f8c8d']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
DIV_BLUE, DIV_GRAY, DIV_RED = '#2a78d6', '#f0efec', '#e34948'   # diverging 蓝↔灰↔红
SEQ_LOW, SEQ_HIGH = '#f0efec', '#2f6f9f'                          # 顺序色(弱→强)
DIVERGING_COLS = ('v_r', 'v_r_comp', 'doppler', 'velocity')       # 这些列用 diverging


def get_ds(cfg_file, split):
    """按 split 构建 eval 模式 dataset,返回 (ds, cfg)。

    注意不能用 set_split:基类 set_split 不重载 infos(eval 构造固定载 'test' 模式
    的 val infos),切 split 后 sample_id_list 与 infos 错位。
    这里直接覆写 DATA_SPLIT.test + INFO_PATH.test 后重建,sample_id_list/infos/
    __getitem__ 三者保证同 split 对齐(推理 GT 才正确)。
    """
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader

    cfg = cfg_from_yaml_file(Path(cfg_file), EasyDict())
    cfg.DATA_CONFIG.DATA_SPLIT.test = split
    if split in ('train', 'trainval'):
        cfg.DATA_CONFIG.INFO_PATH.test = list(cfg.DATA_CONFIG.INFO_PATH.train)
    ds, _, _ = build_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, workers=0,
        training=False, logger=None, dist=False)
    return ds, cfg  # 本仓 build_dataloader 直接返回 dataset 本体


def get_gt(base, sid):
    """从 infos 取 GT(gt_boxes_lidar 已是 lidar 系 (N,7));按 lidar_idx 匹配不依赖顺序。"""
    infos = getattr(base, 'kitti_infos', None) or getattr(base, 'vod_infos', None)
    info = _info_by_id(infos).get(sid, None)
    annos = (info or {}).get('annos', {})
    boxes = annos.get('gt_boxes_lidar', np.zeros((0, 7), dtype=np.float32))
    names = annos.get('name', np.array([]))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 7), [str(n) for n in names]


def _info_by_id(infos):
    """infos list → {lidar_idx: info}。"""
    return {i['point_cloud']['lidar_idx']: i for i in infos}


def split_color(base, feat_names, color_by):
    """确定着色列与色标:速度类 diverging 对称,其余顺序色。"""
    if color_by is None:
        color_by = next((f for f in ('rcs', 'intensity', 'v_r') if f in feat_names),
                        feat_names[-1] if feat_names else 'z')
    col = feat_names.index(color_by) if color_by in feat_names else 2
    diverging = any(d in color_by for d in DIVERGING_COLS)
    return col, color_by, diverging


def plot_frame(base, sid, split, out_dir, feat_names, color_by=None, pred=None,
               class_colors=None, point_range=None):
    """单帧出图(三面板: 相机 | GT | pred, 与 visualize_msr.py 同格式)。
    pred: {sample_id: {'pred_boxes','pred_scores','pred_labels'}} 中本帧的项。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from viz_common import draw_box_bev

    pts = base.get_lidar(sid)
    boxes, names = get_gt(base, sid)
    col, cby, diverging = split_color(base, feat_names, color_by)
    class_colors = class_colors or {}

    # ---- 布局: [相机 | BEV+GT | BEV+pred] ----
    img_file = None
    for ext in ('.png', '.jpg'):
        cand = base.root_split_path / 'image_2' / ('%s%s' % (sid, ext))
        if cand.exists():
            img_file = cand
            break
    if img_file is not None:
        fig, (ax_img, ax_gt, ax_pred) = plt.subplots(
            1, 3, figsize=(20, 6.2), width_ratios=[1.15, 1, 1],
            gridspec_kw={'wspace': 0.25})
        from skimage import io
        ax_img.imshow(io.imread(str(img_file)))
        ax_img.set_title('Camera', fontsize=10, color=INK2, pad=8)
        ax_img.axis('off')
    else:
        fig, (ax_gt, ax_pred) = plt.subplots(
            1, 2, figsize=(14, 6.8), gridspec_kw={'wspace': 0.15})
    panels = [ax_gt, ax_pred]

    # ---- BEV 朝向: x 前(屏幕上) / y 左(屏幕左) ----
    # 数据坐标取 (u,v)=(y,x) + invert_xaxis ⇒ 场景逆时针转 90°, tick 仍是真实 y 值
    if pts.shape[0]:
        c = pts[:, col] if pts.shape[1] > col else pts[:, 0]
        if diverging:
            cmap = LinearSegmentedColormap.from_list('div', [DIV_BLUE, DIV_GRAY, DIV_RED])
            vmax = np.percentile(np.abs(c), 99) or 1.0
            norm = Normalize(vmin=-vmax, vmax=vmax)
        else:
            cmap = LinearSegmentedColormap.from_list('seq', [SEQ_LOW, SEQ_HIGH])
            vmax = np.percentile(np.abs(c), 99) or 1.0
            norm = Normalize(vmin=np.percentile(c, 1), vmax=vmax)
        for ax in panels:  # 两面板各画一份(scatter 不可跨 axes 复用)
            sc = ax.scatter(pts[:, 1], pts[:, 0], c=c, cmap=cmap, norm=norm,
                            s=6, linewidths=0, alpha=0.8, zorder=2)
        cb = fig.colorbar(sc, ax=panels, fraction=0.046, pad=0.02)
        cb.set_label(cby, fontsize=9, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)

    # ---- GT(ax_gt) 与 pred(ax_pred): 同款细实线空心,分面板独立展示 ----
    n_pred = 0
    for b, n in zip(boxes, names):
        draw_box_bev(ax_gt, b, class_colors.get(n, '#eda100'),
                     linewidth=1.2, zorder=4, swap_xy=True)
    if pred is not None and pred.get('pred_boxes') is not None:
        class_names = base.class_names
        for b, lb, sc_ in zip(pred['pred_boxes'], pred['pred_labels'], pred['pred_scores']):
            n = class_names[int(lb) - 1] if 0 < int(lb) <= len(class_names) else str(lb)
            draw_box_bev(ax_pred, b, class_colors.get(n, '#eda100'),
                         linewidth=1.2, zorder=4, swap_xy=True)
            n_pred += 1

    for ax in panels:
        ax.scatter([0], [0], marker='o', s=5, color=INK, zorder=5)
    ax_gt.annotate('ego', (0, 0), textcoords='offset points', xytext=(6, 6),
                   fontsize=8, color=INK2)

    # 范围:点与框联合外沿 + 3m 边距(cfg 的 POINT_CLOUD_RANGE 可覆盖),两面板同 range; 等比
    if point_range is not None:
        x0, y0, _z0, x1, y1, _z1 = point_range
        xlo, xhi, ylo, yhi = x0, x1, y0, y1
    elif pts.shape[0] or boxes.shape[0]:
        xs = np.concatenate([p for p in (pts[:, 0], boxes[:, 0]) if p.size])
        ys = np.concatenate([p for p in (pts[:, 1], boxes[:, 1]) if p.size])
        m = 3.0
        xlo, xhi = min(xs.min(), 0) - m, xs.max() + m
        ylo, yhi = min(ys.min(), 0) - m, ys.max() + m
    else:
        xlo, xhi, ylo, yhi = -10, 10, -10, 10
    for ax in panels:
        ax.set_xlim(ylo, yhi)          # 显示横轴 = y
        ax.set_ylim(xlo, xhi)          # 显示纵轴 = x
        ax.invert_xaxis()              # +y 朝左
        ax.set_aspect('equal')
        ax.set_xlabel('y (m)', fontsize=9, color=INK2)
        ax.set_ylabel('x (m)', fontsize=9, color=INK2)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)

    # 图例: 全类固定槽位色, 两面板各带一份; title 带各自目标数
    class_handles = [Line2D([0], [0], color=class_colors[n], linewidth=2, label=n)
                     for n in class_colors]
    for ax, ttl in zip(panels, ['GT (%d)' % boxes.shape[0], 'Pred (%d)' % n_pred]):
        ax.legend(handles=class_handles, loc='upper right', fontsize=8,
                  framealpha=0.9, edgecolor=GRID, labelcolor=INK2,
                  title='Class', title_fontsize=9)
        ax.set_title(ttl, fontsize=10, color=INK2, pad=8)

    fig.suptitle('%s %s %s  |  %d GT, %d pred'
                 % (Path(base.root_path).name, split, sid, boxes.shape[0], n_pred),
                 fontsize=12, color=INK, y=0.99)

    out = out_dir / ('%s_bev_%s.png' % (split, sid))
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out, pts.shape[0], boxes.shape[0], n_pred


def main():
    parser = argparse.ArgumentParser(description='KITTI/VoD/Astyx BEV visualization (GT + optional pred)')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='模型 cfg(含 _BASE_CONFIG_ 数据集引用,如 experiments/YAML/a1.yaml)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint;不给则纯数据模式(仅 GT)')
    parser.add_argument('--split', type=str, default='both',
                        choices=['train', 'val', 'test', 'both'],
                        help="both=train+val 各抽 num 帧('test' 为 KITTI testing 无 GT,慎用)")
    parser.add_argument('--num', type=int, default=10, help='每个 split 选帧数')
    parser.add_argument('--indices', type=str, nargs='*', default=None,
                        help='显式帧号列表(给出则忽略自动选帧,仍读该 split)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录;默认 output/res_viz/<cfg名>/')
    parser.add_argument('--color_by', type=str, default=None,
                        help='着色列名(used_feature_list 内;默认自动 rcs→intensity→v_r)')
    args = parser.parse_args()

    from viz_common import pick_frames, build_viz_net, infer_sample_ids

    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path('output/res_viz') / Path(args.cfg_file).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 推理网络只建一次(按模型 cfg,split 无关;数据集 per-split 重建)
    net = None
    if args.ckpt is not None:
        net, _, _, _ = build_viz_net(args.cfg_file, args.ckpt)

    for split in splits:
        base, cfg = get_ds(args.cfg_file, split)
        feat_names = list(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
        class_colors = {n: PALETTE[i % len(PALETTE)]
                        for i, n in enumerate(base.class_names)}
        gt_by_id = {}
        for sid in base.sample_id_list:
            gt_by_id[sid] = get_gt(base, sid)
        ids = args.indices if args.indices else pick_frames(gt_by_id, args.num)

        pred_by_id = infer_sample_ids(net, base, ids) if net is not None else {}
        pcr = [float(v) for v in cfg.DATA_CONFIG.POINT_CLOUD_RANGE] \
            if 'POINT_CLOUD_RANGE' in cfg.DATA_CONFIG else None

        print('=== visualize_kitti: split=%s, %d frames ===' % (split, len(ids)))
        for sid in ids:
            out, n_pts, n_gt, n_p = plot_frame(
                base, sid, split, out_dir, feat_names, args.color_by,
                pred=pred_by_id.get(sid), class_colors=class_colors, point_range=pcr)
            print('  %s -> %s  (pts=%d gt=%d pred=%d)' % (sid, out, n_pts, n_gt, n_p))
    print('=== done, output dir: %s ===' % out_dir)


if __name__ == '__main__':
    main()
