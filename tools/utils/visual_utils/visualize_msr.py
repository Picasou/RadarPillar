#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR 多帧点云+GT 可视化脚本。

按 MsrDataset 现有解析策略(struct.json 动态 dtype + POINTS scale + LABELS ×0.01)
读取若干帧,输出 每帧一图:左=相机图像(仅上下文,无有效标定不可投影),中/右=BEV 点云框图。
- 三面板:   [相机 | BEV+GT | BEV+pred],GT 与 pred 分面板独立展示,便于对比
- BEV 朝向: x 前(屏幕上) / y 左(屏幕左),车规惯例
- 点着色:   doppler_gnd(对地多普勒,ego 补偿后地速矢量投影回径向),蓝灰红 diverging,对称色标
- GT 框:    细实线 + 无颜色填充,类别固定槽位色 '1'Car=蓝 '2'Ped=紫 '4'Cyclist=橙 '5'Truck=青
- 预测框:   粗虚线 + 半透明颜色填充;--cfg_file + --ckpt 给出时才有,否则 pred 面板纯点云
- 选帧:     扫 split 内 LABELS,优先类多样性,再按 GT 数;除显式 --split testing 外强制排除测试集帧
- 输出:     默认 output/res_viz/<cfg名>/(--cfg_file 时)或 output/res_viz/msr_data/

用法:
  python tools/utils/visual_utils/visualize_msr.py --split training --num 6
  python tools/utils/visual_utils/visualize_msr.py --indices 00000000 00000500
  python tools/utils/visual_utils/visualize_msr.py --cfg_file experiments/MC_DATASET/YAML/msr_radarpillar.yaml \
      --ckpt output/.../best.pth --split both --num 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根,供 import pcdet
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'scripts' / 'data'))  # 供 import check_msr

CLASS_LABEL = {'1': 'Car', '2': 'Pedestrian', '4': 'Cyclist', '5': 'Truck'}
# 类色避开点云 doppler diverging(蓝↔灰↔红)的色域: 取黄/品红/草绿/青绿,
# 框与点云色相分离, 不再和蓝点/红点混淆
CLASS_COLOR = {'1': '#f1c40f', '2': '#d63ee0', '4': '#7cb342', '5': '#00b3a4'}
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


def plot_frame(ds, sid, split, out_dir, color_by='doppler_gnd', pred=None,
               x_range=None, y_range=None, cbar_range=None, border_black=False,
               tracks=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from viz_common import draw_box_bev

    pts = ds.get_radar(sid)                       # (N, 18) MSR_FEATURE_ORDER 全列
    cols = {n: i for i, n in enumerate(ds.radar_feature_order)}
    boxes, names = ds.get_label(sid)
    param = ds.get_dynamic_param(sid)

    # ---- 布局: [相机 | BEV+GT | BEV+pred] ----
    img_file = ds.root_split_path / 'IMAGES' / ('%s.png' % sid)
    if img_file.exists():
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

    # ---- BEV 朝向: x 前(屏幕上) / y 左(屏幕左) = 车规惯例 ----
    # 实现: 数据坐标取 (u,v)=(y,x), 再 invert_xaxis ⇒ 整个场景逆时针转 90°,
    # 横轴 tick 仍是真实 y 值(左大右小), 无需手写负号
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
        # cbar_range 传入则固定色标(序列动画防逐帧跳变),否则逐帧 p99 自适应
        vmax = cbar_range if cbar_range else (np.percentile(np.abs(c), 99) or 1.0)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        for ax in panels:  # 两个 BEV 面板各画一份(scatter 不能跨 axes 复用)
            sc = ax.scatter(pts[:, 1], pts[:, 0], c=c, cmap=cmap, norm=norm,
                            s=2, linewidths=0, alpha=0.9, zorder=2)
        cb = fig.colorbar(sc, ax=panels, fraction=0.046, pad=0.02)
        cb.set_label(color_by, fontsize=9, color=INK2)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)

    # GT 框(ax_gt): 细实线 + 无颜色填充; 统一走 draw_box_bev(Polygon 角点法)
    for b, n in zip(boxes, names):
        color = CLASS_COLOR.get(str(n), '#eda100')
        draw_box_bev(ax_gt, b, color, linestyle='-', linewidth=1.2, zorder=4,
                     swap_xy=True)

    # 预测框(ax_pred): 与 GT 同款细实线空心,风格统一(区分只靠分面板+各自计数)
    n_pred = 0
    if pred is not None:
        cls_names = ['1', '2', '4', '5']  # MsrDataset CLASS_NAMES(顺序即 label 1/2/3/4 槽位)
        for b, lb, sc_ in zip(pred.get('pred_boxes', []),
                              pred.get('pred_labels', []),
                              pred.get('pred_scores', [])):
            li = int(lb)
            n = cls_names[li - 1] if 0 < li <= len(cls_names) else str(li)
            color = CLASS_COLOR.get(n, '#eda100')
            draw_box_bev(ax_pred, b, color, linestyle='-', linewidth=1.2,
                         zorder=4, swap_xy=True)
            n_pred += 1

    # 航迹框(ax_pred, mod=2 全链路): 同款细实线 + 框旁航迹 ID(与 pred 互斥使用)
    n_trk = 0
    if tracks:
        for b, n, tid in tracks:
            color = CLASS_COLOR.get(str(n), '#eda100')
            draw_box_bev(ax_pred, b, color, linestyle='-', linewidth=1.2,
                         zorder=4, swap_xy=True, label='#%d' % tid)
            n_trk += 1

    # ego 位置
    for ax in panels:
        ax.scatter([0], [0], marker='o', s=5, color=INK, zorder=5)
    ax_gt.annotate('ego', (0, 0), textcoords='offset points', xytext=(6, 6),
                   fontsize=8, color=INK2)

    # 范围:自适应联合外沿 + 3m 边距; x_range/y_range 传入则固定(序列动画防跳变),两面板同 range; 等比
    if pts.shape[0] or boxes.shape[0]:
        xs = np.concatenate([pts[:, 0], boxes[:, 0]]) if pts.shape[0] else boxes[:, 0]
        ys = np.concatenate([pts[:, 1], boxes[:, 1]]) if pts.shape[0] else boxes[:, 1]
        m = 3.0
        xlo, xhi = min(xs.min(), 0) - m, xs.max() + m
        ylo, yhi = min(ys.min(), 0) - m, ys.max() + m
    else:
        xlo, xhi, ylo, yhi = -10, 10, -10, 10
    if x_range:
        xlo, xhi = x_range
    if y_range:
        ylo, yhi = y_range
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

    # 图例: 全 4 类固定槽位色, 面板外共享一份(figure 底部), 不遮挡目标; title 带各自目标数
    class_handles = [Line2D([0], [0], color=CLASS_COLOR[k], linewidth=2,
                            label=CLASS_LABEL.get(k, k))
                     for k in ['1', '2', '4', '5']]
    for ax, ttl in zip(panels, ['GT (%d)' % boxes.shape[0],
                                'Trk (%d)' % n_trk if tracks is not None else 'Pred (%d)' % n_pred]):
        ax.set_title(ttl, fontsize=10, color=INK2, pad=8)
    fig.legend(handles=class_handles, loc='upper center', ncol=4, fontsize=9,
               framealpha=0.9, edgecolor=GRID, labelcolor=INK2,
               title='Class', title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

    n_out = n_trk if tracks is not None else n_pred
    fig.suptitle('MSR %s %s  |  %d GT, %d %s  |  ego %.1f m/s'
                 % (split, sid, boxes.shape[0], n_out,
                    'trk' if tracks is not None else 'pred', param['ego_speed']),
                 fontsize=12, color=INK, y=0.99)

    # border_black: 最后一个面板(BEV+pred)黑色粗边框(split 来源标记);
    # 只动 spines 不改 figure 尺寸,保证序列 PNG 尺寸一致,GIF 合成不异常
    if border_black:
        for s in ax_pred.spines.values():
            s.set_color('black')
            s.set_linewidth(3.5)

    out = out_dir / ('%s_bev_%s.png' % (split, sid))
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return out, pts.shape[0], boxes.shape[0], n_out, sorted(set(str(n) for n in names))


def main():
    parser = argparse.ArgumentParser(description='MSR multi-frame BEV + GT(+pred) visualization')
    parser.add_argument('--data_path', type=str, default='/mnt/d/DataSet/MSRv1')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='模型 cfg;与 --ckpt 连用开启预测叠加模式(如 experiments/MC_DATASET/YAML/msr_radarpillar.yaml)')
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

    net, _loader, _ds, _cfg = build_viz_net(args.cfg_file, args.ckpt) if args.ckpt is not None else (None, None, None, None)

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
                            class_names=['1', '2', '4', '5'], training=False, root_path=None)
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
