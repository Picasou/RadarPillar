#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三套可视化脚本 (msr / kitti / nuscenes) 的共享工具。

- pick_frames:        MSR 式智能选帧(分段覆盖 + 类多样性 + GT 签名去重)
- build_viz_net:      从模型 cfg + ckpt 构建 eval 网络(full_chain viz step 用)
- infer_indices:      指定 dataset index 列表推理,返回 {sample_id: pred_dict}
- draw_box_bev:       BEV 框绘制(GT 实线 / pred 虚线统一入口)

坐标口径: 与 pcdet 一致(x 前 y 左 z 上,heading 绕 z),框 (x,y,z,dx,dy,dz,hdg)。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根


def pick_frames(gt_by_id, num):
    """MSR 式智能选帧。

    帧号排序后均分 num 段(全程覆盖),段内按 类多样性→GT 数 取最优;
    GT 中心点集签名去重,避免重复画面。

    Args:
        gt_by_id: {sample_id: (boxes(N,7), names(list))},boxes 可为 (0,7)
        num:      选帧数
    Returns:
        list[sample_id]
    """
    stats = {}
    for sid, (boxes, names) in gt_by_id.items():
        uniq = set(map(str, names))
        sig = tuple(sorted(map(tuple, np.round(boxes[:, :2], 1)))) if boxes.shape[0] else ()
        stats[sid] = (-len(uniq), -len(names), sig)
    ordered = sorted(stats, key=str)
    n = len(ordered)
    picked, seen = [], set()
    for k in range(num):
        bucket = ordered[k * n // num:(k + 1) * n // num] or ordered[-1:]
        for sid in sorted(bucket, key=lambda s: stats[s][:2]):
            if stats[sid][2] not in seen:
                picked.append(sid)
                seen.add(stats[sid][2])
                break
    return picked


def build_viz_net(cfg_file, ckpt=None):
    """模型 cfg(+ckpt) → (net, loader, ds, cfg)。

    loader/ds 为 eval 模式 batch_size=1;ckpt 给定时加载权重(test.py 同款路径)。
    """
    import torch
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network

    cfg = cfg_from_yaml_file(Path(cfg_file), EasyDict())
    ds, loader, _ = build_dataloader(
        cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, workers=0,
        training=False, logger=None, dist=False)
    net = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    if ckpt is not None:
        state = torch.load(str(ckpt), map_location='cpu')
        net.load_state_dict(state['model_state'], strict=True)
        print('[viz] ckpt loaded: %s (epoch %s)' % (ckpt, state.get('epoch', '?')))
    net.cuda().eval()
    return net, loader, ds, cfg


def infer_indices(net, ds, indices):
    """按 dataset index 推理,返回 {sample_id: pred_dict}。

    pred_dict: {'pred_boxes'(N,7), 'pred_scores'(N,), 'pred_labels'(N,)},
    经模型自带 POST_PROCESSING(NMS/score_thresh)后的最终输出。
    """
    import torch
    from pcdet.models import load_data_to_gpu

    base = ds.dataset if hasattr(ds, 'dataset') else ds  # 兼容 DataLoader wrapper
    out = {}
    with torch.no_grad():
        for idx in indices:
            batch = base.collate_batch([base[int(idx)]])
            load_data_to_gpu(batch)
            pred_dicts, _ = net(batch)
            sid = base.sample_id_list[int(idx)]
            out[sid] = {k: v.detach().cpu().numpy() for k, v in pred_dicts[0].items()
                        if hasattr(v, 'numpy')}
    return out


def infer_sample_ids(net, ds, sample_ids):
    """按 sample_id 推理(内部转 index),返回 {sample_id: pred_dict}。"""
    base = ds.dataset if hasattr(ds, 'dataset') else ds
    id2idx = {sid: i for i, sid in enumerate(base.sample_id_list)}
    indices = [id2idx[sid] for sid in sample_ids]
    return infer_indices(net, ds, indices)


def draw_box_bev(ax, box, color, label=None, linestyle='-', score=None,
                 linewidth=2.0, zorder=4, facecolor='none', alpha=1.0, swap_xy=False):
    """BEV 单框绘制:Polygon 角点法 + 朝向短线(+可选类名/score 文字)。

    box: (7,) [x,y,z,dx,dy,dz,heading];细实线空心用于 GT,粗虚线填充用于 pred。
    角点法直接算 4 角坐标,不依赖 Rectangle 旋转锚点语义,任意朝向精确居中。
    swap_xy=True: 画在 (u,v)=(y,x) 数据坐标(配合 ax.invert_xaxis 实现 x 朝上/y 朝左
    的车规 BEV 朝向),角点与朝向自动随坐标互换。
    朝向短线粗细随 linewidth(GT 细/pred 粗成对);facecolor+alpha 时边线保持不透明、
    仅填充半透明,避免粗虚线被 alpha 洗淡。
    """
    import matplotlib.patheffects as pe
    from matplotlib.patches import Polygon

    x, y, _z, dx, dy, _dz, hdg = box
    fwd = np.array([np.cos(hdg), np.sin(hdg)])   # 车头方向
    left = np.array([-np.sin(hdg), np.cos(hdg)])  # 车身左侧
    c = np.array([x, y])
    corners = np.array([c + fwd * dx / 2 + left * dy / 2,
                        c + fwd * dx / 2 - left * dy / 2,
                        c - fwd * dx / 2 - left * dy / 2,
                        c - fwd * dx / 2 + left * dy / 2])
    tip = c + fwd * dx / 2
    if swap_xy:
        corners, tip, c = corners[:, ::-1], tip[::-1], c[::-1]

    if facecolor not in (None, 'none'):  # 填充层(半透明),边线层单独画保持实色
        ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor,
                             edgecolor='none', alpha=alpha, zorder=zorder - 0.1))
    edge = Polygon(corners, closed=True, linewidth=linewidth, edgecolor=color,
                   facecolor='none', linestyle=linestyle, zorder=zorder)
    edge.set_path_effects([pe.withStroke(linewidth=linewidth + 2, foreground='white')])
    ax.add_patch(edge)
    ax.plot([c[0], tip[0]], [c[1], tip[1]], color=color,
            linewidth=linewidth, linestyle=linestyle, zorder=zorder)  # 朝向(车头)
    if label:
        text = '%s %.2f' % (label, score) if score is not None else label
        ax.text(x, y + max(dy, 1.2) * 0.7 + 0.6, text, fontsize=8,
                color='#0b0b0b', ha='center', zorder=zorder + 1)
        ax.texts[-1].set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
