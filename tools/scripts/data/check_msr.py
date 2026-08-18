#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR 单帧可视化验证脚本(plan Task 6)。

读 POINTS / LABELS / PARAMS bin + 各自 struct.json,打印:
  - POINTS raw dtype(验证 padding 对齐:fields 累加不足则末尾补 _pad)
  - 18 列 features(N, 18) shape 与 dtype
  - LABELS gt_boxes / gt_names
  - PARAMS dynamic_param(ego_speed / yaw_rate)
然后用 matplotlib 画 BEV 散点(x/y)+ 叠加 gt 框(patches.Rectangle),存 png。

参考: tests/msr/test_msr_dataset.py 的 _make_minimal_cfg 构造完整 cfg(含 DATA_PROCESSOR)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from easydict import EasyDict

# 保证 pcdet 可导入(用户运行时显式 PYTHONPATH=tools 也可,这里兜底)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))


def make_full_cfg(data_path):
    """完整 dataset_cfg:含 DATA_PROCESSOR + POINT_FEATURE_ENCODING,供 MsrDataset 构造。

    注意:PointFeatureEncoder 基类硬断言 src_feature_list[0:3] == ['x','y','z']。
    MsrDataset.get_radar 返回 MSR_FEATURE_ORDER 18 列全列(xyz 已在前 3),
    选列由 encoder 按 used_feature_list 统一完成。src_feature_list 直接用
    MSR_FEATURE_ORDER(与 get_radar 输出列序一致,encoder 的 src 索引才正确)。
    """
    return EasyDict({
        'DATASET': 'MsrDataset',
        'DATA_PATH': data_path,
        'DATA_SPLIT': {'train': 'training', 'test': 'val'},
        'INFO_PATH': {'train': ['msr_infos_training.pkl'], 'test': ['msr_infos_val.pkl']},
        'POINT_CLOUD_RANGE': [0, -25.6, -10, 51.2, 25.6, 10],
        'USE_GND_VELOCITY': True,
        'POINT_FEATURE_ENCODING': {
            'encoding_type': 'absolute_coordinates_encoding',
            # 重排:xyz 在前 3(满足 PointFeatureEncoder 断言),其余按 MSR_FEATURE_ORDER 原顺序
            'src_feature_list': [
                'x', 'y', 'z',
                'range_m', 'doppler_mps', 'ang_rad', 'elv_rad', 'rcs', 'snr',
                'doppler_anti_amb_confi', 'exist_confidence', 'frame', 'beam', 'extra_cnt',
                'dop_x', 'dop_y', 'dop_x_gnd', 'dop_y_gnd',
            ],
            # 选 18 列全集以便打印验证;selected_feature_idx 会把 xyz 提到前 3
            'used_feature_list': [
                'x', 'y', 'z', 'dop_x_gnd', 'dop_y_gnd', 'rcs',
                'range_m', 'doppler_mps', 'ang_rad', 'elv_rad', 'snr',
                'doppler_anti_amb_confi', 'exist_confidence', 'frame', 'beam', 'extra_cnt',
                'dop_x', 'dop_y',
            ],
        },
        'POINT_FEATURE_NORMALIZATION': {'USE_NORM': False},
        'DATA_AUGMENTOR': {'DISABLE_AUG_LIST': ['placeholder'], 'AUG_CONFIG_LIST': []},
        'DATA_PROCESSOR': [
            {'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True},
            {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}},
            {'NAME': 'transform_points_to_voxels',
             'VOXEL_SIZE': [0.16, 0.16, 20.0],
             'MAX_POINTS_PER_VOXEL': 10,
             'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}},
        ],
    })


def main():
    parser = argparse.ArgumentParser(description='MSR single-frame sanity check + BEV plot')
    parser.add_argument('--data_path', type=str, default='/mnt/d/DataSet/MSR')
    parser.add_argument('--idx', type=str, default='00000000')
    parser.add_argument('--out_png', type=str, default='/tmp/msr_bev_%s.png',
                        help='output BEV png path (%%s for idx)')
    args = parser.parse_args()

    from pcdet.datasets.msr.msr_dataset import MsrDataset
    from pcdet.datasets.msr.msr_utils import load_struct_dtype

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        print('ERROR: data_path %s 不存在' % data_path, file=sys.stderr)
        sys.exit(1)

    # ---- 直接读 bin + struct.json 打印 raw dtype(验证 padding)----
    print('=' * 64)
    print('1. POINTS raw dtype')
    pts_dtype, pts_total, pts_names = load_struct_dtype(data_path / 'POINTS' / 'struct.json')
    print('  fields      :', pts_names)
    print('  itemsize    :', pts_dtype.itemsize, '(json total_size:', pts_total, ')')
    print('  has _pad    :', '_pad' in pts_dtype.names)
    raw_pts = np.fromfile(str(data_path / 'POINTS' / ('%s.bin' % args.idx)), dtype=pts_dtype)
    print('  N points    :', raw_pts.shape[0])

    print('=' * 64)
    print('2. LABELS raw dtype')
    lbl_dtype, lbl_total, lbl_names = load_struct_dtype(data_path / 'LABELS' / 'struct.json')
    print('  fields      :', lbl_names)
    print('  itemsize    :', lbl_dtype.itemsize, '(json total_size:', lbl_total, ')')
    print('  has _pad    :', '_pad' in lbl_dtype.names, '(28B 预期有 1B padding)')
    raw_lbl = np.fromfile(str(data_path / 'LABELS' / ('%s.bin' % args.idx)), dtype=lbl_dtype)
    print('  N gt        :', raw_lbl.shape[0])

    print('=' * 64)
    print('3. PARAMS raw dtype')
    prm_dtype, prm_total, prm_names = load_struct_dtype(data_path / 'PARAMS' / 'radar_dynamic.struct.json')
    print('  fields      :', prm_names)
    print('  itemsize    :', prm_dtype.itemsize, '(json total_size:', prm_total, ')')

    # ---- 用 MsrDataset 类读完整 18 列 features + label + param ----
    print('=' * 64)
    print('4. MsrDataset reader output')
    ds = MsrDataset(
        dataset_cfg=make_full_cfg(str(data_path)),
        class_names=['1', '4', '5'],
        training=False, root_path=None,
    )
    points = ds.get_radar(args.idx)
    print('  get_radar   : shape=%s dtype=%s' % (points.shape, points.dtype))
    print('  selected    :', ds.selected_feature_list)
    print('  selected_idx:', ds.selected_feature_idx)
    if points.shape[0] > 0:
        print('  x range     : [%.2f, %.2f]' % (points[:, 0].min(), points[:, 0].max()))
        print('  y range     : [%.2f, %.2f]' % (points[:, 1].min(), points[:, 1].max()))

    gt_boxes, gt_names = ds.get_label(args.idx)
    print('  gt_boxes    : shape=%s' % (gt_boxes.shape,))
    print('  gt_names    :', list(gt_names))

    param = ds.get_dynamic_param(args.idx)
    print('  dynamic_param:', param)

    # ---- BEV 可视化 ----
    print('=' * 64)
    print('5. BEV plot')
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无显示环境也能存盘
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print('  matplotlib 未安装,跳过 BEV 图')
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    if points.shape[0] > 0:
        # 散点:x/y,颜色按 rcs
        sc = ax.scatter(points[:, 0], points[:, 1], c=points[:, 5] if points.shape[1] > 5 else 'b',
                        s=4, cmap='viridis', alpha=0.6, label='radar points')
        plt.colorbar(sc, ax=ax, label='rcs' if points.shape[1] > 5 else '')

    # 叠加 gt 框(BEV:用 dx, dy 和 heading 画 Rectangle)
    cmap = {'1': 'red', '4': 'blue', '5': 'green'}
    for i in range(gt_boxes.shape[0]):
        x, y, _z, dx, dy, _dz, heading = gt_boxes[i]
        # OpenPCDet 约定 heading 绕 z 轴;BEV 框中心 (x,y),尺寸 (dx, dy)
        # matplotlib Rectangle 角度逆时针;heading 是绕 z 正方向(逆时针向上为正)
        rect = Rectangle(
            (x - dx / 2, y - dy / 2), dx, dy,
            angle=np.degrees(heading),
            linewidth=1.5, edgecolor=cmap.get(str(gt_names[i]), 'yellow'),
            facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(x, y, str(gt_names[i]), fontsize=9, color=cmap.get(str(gt_names[i]), 'yellow'))

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('MSR BEV idx=%s  (N_pts=%d, N_gt=%d)' % (args.idx, points.shape[0], gt_boxes.shape[0]))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    # legend for classes present
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, label='class %s' % k) for k, c in cmap.items()
               if k in set(str(n) for n in gt_names)]
    if handles:
        ax.legend(handles=handles, loc='upper right')

    out_path = args.out_png % args.idx
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print('  saved BEV png ->', out_path)


if __name__ == '__main__':
    main()
