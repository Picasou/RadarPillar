#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
泛化集连续 mAP 评测: model_1 未训练序列 × 多模型, 复用训练侧 eval 内核。

链路: gt_radar bin(27B gt_radar_target, 与训练 LABELS 同构) → GT annos;
      tracker loader 解点 + RawDetector(不滤分数) → det annos;
      get_msr_eval_result(与 test.py 同内核同阈值) → 总体 + 分序列 mAP。

用法(仓库根): PYTHONPATH=tools python tools/scripts/test/eval_generalization.py \
    [--data_root ...] [--models tag1,tag2|all] [--report ...]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from easydict import EasyDict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from pcdet.config import cfg as pcdet_global_cfg, cfg_from_yaml_file
from pcdet.datasets.msr.msr_dataset import MSR_CLASS_LABEL
from pcdet.datasets.msr.msr_utils import boxes_lidar_to_pseudo_camera
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from tracker.detector import Detector
from tracker.loader import Loader
from tracker.utils.common import c_points_prepare
from tracker.utils.rw_struct import struct_read, Raw_TrkHead

# 本次泛化实验的模型清单: tag → run 目录名 (output/train_log/msr/ 下, cfg/best.pth 取自 resbag/)
MODEL_RUNS = {
    'msr_radarpillar':        '202608181912_rpillar_msr_msr',
    'msr_radarNeXt':          '202608221828_msr_radarNeXt_msr_radarNeXt',
    'msr_centerhead2d_trunk': '202608220357_rpillar_msr_centerhead2d_trunk_msr_centerhead2d_trunk',
    'msr_pp64_ch2d_trunk':    '202608220918_rpillar_msr_pp64_ch2d_trunk_msr_pp64_ch2d_trunk',
    'msr_ppmix_ch2d_trunk':   '202608221247_rpillar_msr_ppmix_ch2d_trunk_msr_ppmix_ch2d_trunk',
    'msr_pp333mix_ch2d_trunk': '202608231928_rpillar_msr_pp333mix_ch2d_trunk_msr_pp333mix_ch2d_trunk',
}
RESBAG_ROOT = ROOT / 'output' / 'train_log' / 'msr'
# 点云 range 预滤(对齐 tracker 链路; detector prepare_data 内还会再 mask 一次)
MSR_PCR = [0, -20.0, -10, 200, 20.0, 10]
CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Truck']


class RawDetector(Detector):
    """
    检测原生生版: 输出未滤分数的 pred boxes/scores/labels, 供 mAP anno 组装
    """

    def run_raw(self, points: np.ndarray, vdd) -> dict:
        if points is None or points.shape[0] == 0:
            return {'boxes': np.zeros((0, 7), np.float32),
                    'scores': np.zeros(0, np.float32),
                    'labels': np.zeros(0, np.int64)}
        data_dict = self._prepare(self._to_src_points(points, vdd))
        pred = self._infer(data_dict)[0]
        return {
            'boxes': pred['pred_boxes'].cpu().numpy(),
            'scores': pred['pred_scores'].cpu().numpy(),
            'labels': pred['pred_labels'].cpu().numpy(),
        }


def load_gt_seq(seq_path: str) -> list:
    """
    GT 加载: gt_radar 流 → 逐帧 (gt_boxes(N,7)[x,y,z,l,w,h,heading_rad], names 语义类名)
    布局 = LABELS struct.json 27B gt_radar_target: id + i16×9{x,y,z,vx,vy,heading_deg,w,l,h}(×0.01) + u8×7
    """
    g = Path(seq_path) / 'gt.default'
    heads = struct_read(str(g / 'gt_radar_1200.00000.bin'), Raw_TrkHead)
    data = (g / 'gt_radar_1201.00000.bin').read_bytes()
    rec_n = sum(h.trk_num for h in heads)
    stride = len(data) / rec_n if rec_n else 27
    assert stride in (27.0, 28.0) and rec_n * int(stride) == len(data), \
        'GT 记录布局非 27/28B: %s (%.2f B/rec)' % (seq_path, stride)
    stride = int(stride)   # 28B = DATASETv1 旧格式(27B + 尾部 1B pad)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(rec_n, stride)[:, :27]
    i10 = raw[:, :20].copy().view('<i2').reshape(rec_n, 10).astype(np.float64)  # id,x,y,z,vx,vy,head,w,l,h
    v = i10[:, 1:] * 0.01                       # x,y,z,vx,vy,heading_deg,w,l,h (物理量)
    types = raw[:, 20].copy()

    frames, off = [], 0
    for h in heads:
        n = h.trk_num
        x, y, z, _, _, head, w, l, hgt = v[off:off + n].T
        boxes = np.stack([x, y, z, l, w, hgt, head * np.pi / 180.0], axis=1).astype(np.float32)
        names = np.array([MSR_CLASS_LABEL.get(str(int(t)), str(int(t))) for t in types[off:off + n]])
        frames.append((boxes, names))
        off += n
    return frames


def make_anno(boxes: np.ndarray, names: np.ndarray, scores=None) -> dict:
    """
    anno 组装: GT 仿 MsrDataset.evaluation / det 仿 generate_prediction_dicts (alpha=-10 关 AOS, bbox=0)
    """
    anno = {
        'name': np.asarray(names),
        'alpha': np.full(len(names), -10.),
        'bbox': np.zeros([len(names), 4]),
        'score': np.ones(len(names)) if scores is None else np.asarray(scores, np.float64),
    }
    if len(boxes) > 0:
        location, dimensions, rotation_y = boxes_lidar_to_pseudo_camera(boxes)
        anno['location'], anno['dimensions'], anno['rotation_y'] = location, dimensions, rotation_y
    else:
        anno['location'] = np.zeros([0, 3])
        anno['dimensions'] = np.zeros([0, 3])
        anno['rotation_y'] = np.zeros(0)
    return anno


def _mk_detector_cfg(run_dir: Path) -> EasyDict:
    """
    构造 Detector 所需最小 cfg (仅 MODEL 段)
    """
    c = EasyDict()
    c.MODEL = EasyDict(cfg=str(run_dir / 'cfg.yaml'), ckpt=str(run_dir / 'best.pth'),
                       score_thresh=0.3, device='cuda')
    return c


def cache_sequences(loader: Loader, seq_dirs: list) -> dict:
    """
    点云解析缓存: loader 单次解析全部序列 (模型无关), 供多模型复用
    """
    cached = {}
    for seq_path in seq_dirs:
        t0 = time.time()
        frames = loader.getframes(seq_path)
        vds = loader.getvds(seq_path)
        pts_list = [c_points_prepare(frames, i, vds, 1, MSR_PCR) for i in range(frames.num)]
        vdd_list = [fr.vdd for fr in frames.Lst]
        cached[seq_path] = (pts_list, vdd_list)
        print('  cached %-42s %4d frames (%.0fs)' % (Path(seq_path).name, frames.num, time.time() - t0))
    return cached


def _fmt_ap(label: str, ap: dict, gt_hist: dict = None) -> str:
    """
    报告行: 各类 bev/3d moderate_R40 (%) + 有样本类均值 (无 GT 类显示 · 并不计入)
    """
    parts = []
    vals_b, vals_3 = [], []
    for c in CLASSES:
        if gt_hist is not None and gt_hist.get(c, 0) == 0:
            parts.append('%s ·' % c[:3])
            continue
        b, t3 = ap.get('%s_bev/moderate_R40' % c, float('nan')), ap.get('%s_3d/moderate_R40' % c, float('nan'))
        parts.append('%s %.1f/%.1f' % (c[:3], max(b, 0), max(t3, 0)))
        vals_b.append(max(b, 0)); vals_3.append(max(t3, 0))
    parts.append('**mAP bev %.1f / 3d %.1f**' % (np.mean(vals_b), np.mean(vals_3)))
    return '%s: %s' % (label, ' | '.join(parts))


def eval_one(tag: str, seq_dirs: list, cached: dict, report_lines: list) -> None:
    """
    单模型评测: 逐序列推理 → (总体+分序列) get_msr_eval_result → 报告行
    """
    run_dir = RESBAG_ROOT / MODEL_RUNS[tag] / 'resbag'
    pcdet_cfg = cfg_from_yaml_file(str(run_dir / 'cfg.yaml'), pcdet_global_cfg)
    class_names = list(pcdet_cfg.CLASS_NAMES)
    label_to_name = np.array([MSR_CLASS_LABEL[n] for n in class_names])
    det = RawDetector(_mk_detector_cfg(run_dir))

    t0 = time.time()
    seq_rows = []
    all_gt, all_det = [], []
    for seq_path in seq_dirs:
        seq_name = Path(seq_path).name
        pts_list, vdd_list = cached[seq_path]
        gt_frames = load_gt_seq(seq_path)
        assert len(gt_frames) == len(pts_list), 'GT/点云帧数不齐: %s' % seq_name

        gt_annos, det_annos = [], []
        for i, (points, vdd) in enumerate(zip(pts_list, vdd_list)):
            pred = det.run_raw(points, vdd)
            gt_boxes, gt_names = gt_frames[i]
            gt_annos.append(make_anno(gt_boxes, gt_names))
            det_names = label_to_name[pred['labels'] - 1] if len(pred['labels']) else np.zeros(0)
            det_annos.append(make_anno(pred['boxes'], det_names, pred['scores']))
        all_gt += gt_annos
        all_det += det_annos

        _, ap_seq = kitti_eval.get_msr_eval_result(gt_annos, det_annos, class_names)
        gt_hist = {c: int(sum((a['name'] == c).sum() for a in gt_annos)) for c in CLASSES}
        gt_note = ' '.join('%s:%d' % (c[:3], n) for c, n in gt_hist.items() if n)
        seq_rows.append(_fmt_ap('  %s [%d帧 GT %s]' % (seq_name, len(gt_annos), gt_note), ap_seq, gt_hist))

    _, ap_all = kitti_eval.get_msr_eval_result(all_gt, all_det, class_names)
    gt_hist_all = {c: int(sum((a['name'] == c).sum() for a in all_gt)) for c in CLASSES}
    report_lines.append('\n## %s  (%.0fs, GT %s)\n' % (
        tag, time.time() - t0, ' '.join('%s:%d' % (c[:3], n) for c, n in gt_hist_all.items() if n)))
    report_lines.append(_fmt_ap('总体', ap_all, gt_hist_all))
    report_lines.extend(seq_rows)
    print('  [%s] %.0fs done' % (tag, time.time() - t0))


def main():
    parser = argparse.ArgumentParser(description='generalization mAP eval on model_1 sequences')
    parser.add_argument('--data_root', type=str, default='/mnt/d/DataSet/.generalization/model_1')
    parser.add_argument('--models', type=str, default='all', help='逗号分隔 tag 或 all')
    parser.add_argument('--report', type=str, default='output/generalization_eval_report.md')
    parser.add_argument('--append', action='store_true', help='报告追加模式(每模型独立进程跑时避免重复表头)')
    args = parser.parse_args()

    seq_dirs = sorted(str(p) for p in Path(args.data_root).iterdir() if p.is_dir())
    tags = list(MODEL_RUNS) if args.models == 'all' else [t.strip() for t in args.models.split(',')]
    print('=== eval_generalization: %d seqs x %d models ===' % (len(seq_dirs), len(tags)))

    print('--- 解析点云(缓存一次) ---')
    cached = cache_sequences(Loader(EasyDict()), seq_dirs)

    appending = args.append and Path(args.report).exists()
    report_lines = [] if appending else [
        '# 泛化集 mAP 评测 (%s, %d 序列)' % (Path(args.data_root).name, len(seq_dirs)),
        '- 数据: %s' % args.data_root,
        '- 口径: get_msr_eval_result (Car/Truck BEV0.5·3D0.25, Ped/Cyc 0.25/0.25), '
        'det 不滤分数; 各类为 bev/3d moderate_R40 (%)',
        '- GT: gt_radar 流 (z=0 BEV, 与训练 LABELS 同构同口径)']
    for tag in tags:
        print('--- %s ---' % tag)
        eval_one(tag, seq_dirs, cached, report_lines)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'a' if appending else 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    print('=== report -> %s ===' % args.report)


if __name__ == '__main__':
    main()
