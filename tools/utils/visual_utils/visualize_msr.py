#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR 多帧点云+GT 可视化脚本。

按 MsrDataset 现有解析策略(struct.json 动态 dtype + POINTS scale + LABELS ×0.01)
读取若干帧,输出 每帧一图:左=相机图像(仅上下文,无有效标定不可投影),右=BEV 点云+GT 框。
- 点着色: doppler_gnd(对地多普勒,ego 补偿后地速矢量投影回径向),蓝灰红 diverging,对称色标
- GT 框:   类别固定槽位色 '1'Car=蓝 '4'Cyclist=橙 '5'Truck=青,框上直接标类名(不单靠颜色)
- 预测框:  --cfg_file + --ckpt 给出时叠加(同类同色,虚线+score);否则纯数据模式
- 选帧:    扫 split 内 LABELS,优先类多样性,再按 GT 数;除显式 --split testing 外强制排除测试集帧
- 输出:    默认 output/res_viz/<cfg名>/(--cfg_file 时)或 output/res_viz/msr_data/

用法:
  python tools/utils/visual_utils/visualize_msr.py --split training --num 6
  python tools/utils/visual_utils/visualize_msr.py --indices 00000000 00000500
  python tools/utils/visual_utils/visualize_msr.py --cfg_file mc/YAML/msr_radarpillar.yaml \
      --ckpt output/.../best.pth --split both --num 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根,供 import pcdet
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'scripts' / 'data'))  # 供 import check_msr

CLASS_LABEL = {'1': 'Car', '4': 'Cyclist', '5': 'Truck'}
CLASS_COLOR = {'1': '#2a78d6', '4': '#eb6834', '5': '#1baf7a'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
DIV_BLUE, DIV_GRAY, DIV_RED = '#2a78d6', '#f0efec', '#e34948'  # diverging 蓝↔灰↔红


def pick_frames(ds, ids, num, split='training'):
    """
    选帧:帧号排序后均分 num 段(全程覆盖),段内按 类多样性→GT 数 取最优;
    GT 框签名(中心点集)去重,避免重复画面。
    split!='testing' 时强制剔除 testing.txt 内帧(不看测试集 GT,防信息泄露)。
    """
    if split != 'testing':
        testing_file = ds.root_path / 'IMAGESETS' / 'testing.txt'
        if testing_file.exists():
            testing = set(x.strip() for x in open(testing_file).readlines())
            ids = [i for i in ids if i not in testing]
    from viz_common import pick_frames as _pick
    gt_by_id = {sid: ds.get_label(sid) for sid in ids}
    return _pick(gt_by_id, num)


def plot_frame(ds, sid, split, out_dir, color_by='doppler_gnd', pred=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    pts = ds.get_radar(sid)                       # (N, 18) MSR_FEATURE_ORDER 全列
    cols = {n: i for i, n in enumerate(ds.radar_feature_order)}
    boxes, names = ds.get_label(sid)
    param = ds.get_dynamic_param(sid)

    # ---- 左:相机图 ----
    img_file = ds.root_split_path / 'IMAGES' / ('%s.png' % sid)
    if img_file.exists():
        fig, (ax_img, ax) = plt.subplots(
            1, 2, figsize=(16, 6.2), width_ratios=[1.25, 1],
            gridspec_kw={'wspace': 0.08})
        from skimage import io
        ax_img.imshow(io.imread(str(img_file)))
        ax_img.set_title('Camera %s (context only, no valid calib)' % sid,
                         fontsize=10, color=INK2, pad=8)
        ax_img.axis('off')
    else:
        fig, ax = plt.subplots(figsize=(9, 7))

    # ---- 右:BEV ----
    cmap = LinearSegmentedColormap.from_list('msr_div', [DIV_BLUE, DIV_GRAY, DIV_RED])
    if pts.shape[0]:
        if color_by == 'doppler_gnd':
            # 对地径向多普勒:ego 补偿后地速矢量(dop_x_gnd,dop_y_gnd)投影回径向单位向量;
            # ego_speed=0 时恒等于 doppler_mps,ego≠0 时自动去除自车运动贡献,保留正负
            azi = pts[:, cols['ang_rad']]
            c = (pts[:, cols['dop_x_gnd']] * np.cos(azi)
                 + pts[:, cols['dop_y_gnd']] * np.sin(azi))
        else:
            c = pts[:, cols[color_by]]
        vmax = np.percentile(np.abs(c), 99) or 1.0
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap=cmap,
                        norm=Normalize(vmin=-vmax, vmax=vmax),
                        s=14, linewidths=0, alpha=0.9, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(color_by, fontsize=9, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)

    # GT 框 + 类名直接标注 + 朝向短线
    for b, n in zip(boxes, names):
        n = str(n)
        x, y, _z, dx, dy, _dz, hdg = b
        color = CLASS_COLOR.get(n, '#eda100')
        rect = Rectangle((x - dx / 2, y - dy / 2), dx, dy,
                         angle=np.degrees(hdg), linewidth=2,
                         edgecolor=color, facecolor='none', zorder=4)
        rect.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])
        ax.add_patch(rect)
        ax.plot([x, x + dx / 2 * np.cos(hdg)], [y, y + dx / 2 * np.sin(hdg)],
                color=color, linewidth=1.2, zorder=4)  # 朝向(车头)
        label = CLASS_LABEL.get(n, n)
        ax.text(x, y + max(dy, 1.2) * 0.7 + 0.6, label, fontsize=8, color=INK,
                ha='center', zorder=5)
        ax.texts[-1].set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    # 预测框(虚线 + score;同类同色,与 GT 虚实区分)
    n_pred = 0
    if pred is not None:
        from viz_common import draw_box_bev
        cls_names = ['1', '4', '5']  # MsrDataset CLASS_NAMES(顺序即 label 1/2/3 槽位)
        for b, lb, sc_ in zip(pred.get('pred_boxes', []),
                              pred.get('pred_labels', []),
                              pred.get('pred_scores', [])):
            li = int(lb)
            n = cls_names[li - 1] if 0 < li <= len(cls_names) else str(li)
            color = CLASS_COLOR.get(n, '#eda100')
            draw_box_bev(ax, b, color, label=CLASS_LABEL.get(n, n),
                         linestyle='--', score=float(sc_), linewidth=1.5, zorder=3.5)
            n_pred += 1

    # ego 位置
    ax.scatter([0], [0], marker='o', s=5, color=INK, zorder=5)
    ax.annotate('ego', (0, 0), textcoords='offset points', xytext=(6, 6),
                fontsize=8, color=INK2)

    # 范围:点与框联合外沿 + 3m 边距,等比
    if pts.shape[0] or boxes.shape[0]:
        xs = np.concatenate([pts[:, 0], boxes[:, 0]]) if pts.shape[0] else boxes[:, 0]
        ys = np.concatenate([pts[:, 1], boxes[:, 1]]) if pts.shape[0] else boxes[:, 1]
        m = 3.0
        ax.set_xlim(min(xs.min(), 0) - m, xs.max() + m)
        ax.set_ylim(min(ys.min(), 0) - m, ys.max() + m)
    ax.set_aspect('equal')

    ax.set_xlabel('x (m)', fontsize=9, color=INK2)
    ax.set_ylabel('y (m)', fontsize=9, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)

    present = [k for k in CLASS_COLOR if k in set(str(n) for n in names)]
    handles = [Line2D([0], [0], color=CLASS_COLOR[k], linewidth=2,
                      label=CLASS_LABEL.get(k, k)) for k in present]
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9,
                  edgecolor=GRID, labelcolor=INK2)

    ax.set_title('BEV radar points + GT(solid) + pred(dashed)  (color = %s)' % color_by,
                 fontsize=10, color=INK2, pad=8)
    fig.suptitle('MSR %s  idx=%s   |   %d points, %d GT, %d pred   |   ego_speed %.1f m/s'
                 % (split, sid, pts.shape[0], boxes.shape[0], n_pred, param['ego_speed']),
                 fontsize=12, color=INK, y=0.99)

    out = out_dir / ('%s_bev_%s.png' % (split, sid))
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out, pts.shape[0], boxes.shape[0], n_pred, sorted(set(str(n) for n in names))


def main():
    parser = argparse.ArgumentParser(description='MSR multi-frame BEV + GT(+pred) visualization')
    parser.add_argument('--data_path', type=str, default='/mnt/d/DataSet/MSR')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='模型 cfg;与 --ckpt 连用开启预测叠加模式(如 mc/YAML/msr_radarpillar.yaml)')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint(best.pth)')
    parser.add_argument('--split', type=str, default='training',
                        choices=['training', 'val', 'testing', 'both'],
                        help='both=training+val 各抽 num 帧')
    parser.add_argument('--num', type=int, default=6, help='每个 split 自动选帧数量')
    parser.add_argument('--indices', type=str, nargs='*', default=None,
                        help='显式帧号列表(给出则忽略 --split 自动选帧,但仍读该 split)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录;默认 output/res_viz/<cfg名>/(--cfg_file 时)或 output/res_viz/msr_data/')
    parser.add_argument('--color_by', type=str, default='doppler_gnd',
                        choices=['doppler_gnd', 'doppler_mps', 'rcs', 'z', 'range_m'])
    args = parser.parse_args()

    from pcdet.datasets.msr.msr_dataset import MsrDataset
    from check_msr import make_full_cfg
    from viz_common import build_viz_net, infer_sample_ids

    splits = ['training', 'val'] if args.split == 'both' else [args.split]
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path('output/res_viz') / (Path(args.cfg_file).stem if args.cfg_file else 'msr_data'))
    out_dir.mkdir(parents=True, exist_ok=True)

    net = build_viz_net(args.cfg_file, args.ckpt) if args.ckpt is not None else None

    for split in splits:
        if args.cfg_file is not None:
            # 预测叠加模式:按模型 cfg 构造,覆写 test split 保证 sample_id_list/
            # infos/__getitem__ 同 split 对齐(set_split 不重载 infos,直接切会错位)
            from easydict import EasyDict
            from pcdet.config import cfg_from_yaml_file
            mcfg = cfg_from_yaml_file(Path(args.cfg_file), EasyDict())
            mcfg.DATA_CONFIG.DATA_SPLIT.test = split
            if split == 'training':
                mcfg.DATA_CONFIG.INFO_PATH.test = list(mcfg.DATA_CONFIG.INFO_PATH.train)
            ds = MsrDataset(dataset_cfg=mcfg.DATA_CONFIG, class_names=list(mcfg.CLASS_NAMES),
                            training=False, root_path=None)
        else:
            ds = MsrDataset(dataset_cfg=make_full_cfg(str(Path(args.data_path))),
                            class_names=['1', '4', '5'], training=False, root_path=None)
            ds.set_split(split)

        ids = args.indices if args.indices else pick_frames(ds, ds.sample_id_list, args.num,
                                                            split=split)
        pred_by_id = infer_sample_ids(net, ds, ids) if net is not None else {}

        print('=== plot_msr_samples: split=%s, %d frames ===' % (split, len(ids)))
        for sid in ids:
            out, n_pts, n_gt, n_p, cls = plot_frame(ds, sid, split, out_dir, args.color_by,
                                                    pred=pred_by_id.get(sid))
            print('  %s -> %s  (pts=%d gt=%d pred=%d classes=%s)' % (sid, out, n_pts, n_gt, n_p, cls))
    print('=== done, output dir: %s ===' % out_dir)


if __name__ == '__main__':
    main()
