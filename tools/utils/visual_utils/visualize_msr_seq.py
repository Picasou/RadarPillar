#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR 连续帧逐帧推理可视化。

MSR 由连续录制切帧而来,帧号即时间序。检测环节仿 tracker/detector.py 的内存点云
直推方式,不走 infos/dataloader:
  POINTS/{sid}.bin → get_radar (N,18) 全列 → DatasetTemplate.prepare_data
  (完整 encoder + DATA_PROCESSOR 管线) → collate → net.forward → pred
出图复用 visualize_msr.plot_frame 现有三面板逻辑(Camera | BEV+GT | BEV+pred),
可视化格式与 visualize_msr.py 完全一致。

--save/--overlap:
  save 四档: none=不落盘 / png=逐帧PNG / gif=PNG+自动合成GIF(960宽)
             / mp4=PNG+自动合成MP4(1440宽 CRF18 高画质,体积远小于GIF);
  save 默认 png,动画档显式指定;overlap=1 覆盖已存在 / 0 跳过(默认 1)

- 连续区间:  --start_sid 起连续 --num 帧(可 --stride 抽稀),按帧号推进
- 固定范围:  BEV 固定 x[-10,210] y[-25,25](ROI 外留边),防序列视角跳变; --x_range/--y_range 可改
- 固定色标:  点云热图默认 ±16 m/s(抽样 p99),防逐帧重归一; --cbar_range 可改(0=逐帧自适应)
- 泄露防护:  区间内 testing 帧默认跳过;--allow_testing 显式放行
- split 标记: val/testing 帧最后一个面板(BEV+pred)加黑色边框(training 无框),不影响画布尺寸
- 空帧:      0 点帧跳过推理,pred 面板纯点云
- 输出:      默认 output/res_viz_seq/<cfg名>/(从仓库根运行)

用法(仓库根):
  python tools/utils/visual_utils/visualize_msr_seq.py --start_sid 00000000 --num 30
  python tools/utils/visual_utils/visualize_msr_seq.py --start_sid 00000410 --num 60 --stride 2
  python tools/utils/visual_utils/visualize_msr_seq.py --save mp4 --start_sid 00000410 --num 40
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo 根,供 import pcdet / tracker
sys.path.insert(0, str(Path(__file__).resolve().parent))      # 本目录,供 import viz_common / visualize_msr


def save_animation(out_dir, fmt):
    """
    序列动画合成: out_dir 内 seq_bev_*.png 按帧号序 → gif(960 宽) 或 mp4(1440 宽 CRF18 高画质)
    """
    import glob

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    src = sorted(glob.glob(str(Path(out_dir) / 'seq_bev_*.png')))
    if not src:
        print('  [anim] no seq_bev_*.png in %s, skip' % out_dir)
        return None
    W = 960 if fmt == 'gif' else 1440
    # 两遍扫描控内存: 第一遍只读尺寸定统一画布高(白底补齐,消逐帧微差;H 取偶兼容 yuv420p)
    hs = []
    for f in src:
        w, h = Image.open(f).size
        hs.append(round(h * W / w))
    H = max(hs)
    H += H % 2
    out = Path(out_dir) / ('msr_seq.%s' % fmt)
    kw = (dict(mode='I', duration=125, loop=0) if fmt == 'gif' else
          dict(fps=8, codec='libx264', macro_block_size=1,
               ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '18']))
    writer = imageio.get_writer(out, **kw)
    for f, h in zip(src, hs):
        img = Image.open(f).convert('RGB').resize((W, h), Image.LANCZOS)
        canvas = Image.new('RGB', (W, H), 'white')
        canvas.paste(img, (0, (H - h) // 2))
        writer.append_data(np.asarray(canvas))
    writer.close()
    print('  [anim] %s (%d frames, %dx%d)' % (out.name, len(src), W, H))
    return out


def main():
    parser = argparse.ArgumentParser(description='MSR sequential per-frame inference + visualization')
    parser.add_argument('--cfg_file', type=str, default='store/202608181912_rpillar_msr_msr/msr.yaml',
                        help='模型 yaml(默认 resbag 里的 MSR RadarPillar)')
    parser.add_argument('--ckpt', type=str, default='store/202608181912_rpillar_msr_msr/best.pth',
                        help='checkpoint(默认 resbag best.pth)')
    parser.add_argument('--start_sid', type=str, default='00000000',
                        help='起始帧号(含),帧号即时间序,如 00000410')
    parser.add_argument('--num', type=int, default=30, help='连续帧数(按 stride 推进)')
    parser.add_argument('--stride', type=int, default=1, help='帧步进(抽稀)')
    parser.add_argument('--allow_testing', action='store_true',
                        help='放行区间内 testing 帧(默认跳过,防信息泄露)')
    parser.add_argument('--save', type=str, choices=['none', 'png', 'gif', 'mp4'], default='png',
                        help='保存方式: none=不落盘 png=逐帧PNG gif=PNG+合成GIF mp4=PNG+合成MP4'
                             '(1440宽高画质);默认 png')
    parser.add_argument('--overlap', type=int, choices=[0, 1], default=1,
                        help='覆盖开关(0=已存在跳过 1=覆盖);默认 1')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录;默认 output/res_viz_seq/<cfg名>/')
    parser.add_argument('--color_by', type=str, default='doppler_gnd',
                        choices=['doppler_gnd', 'doppler_mps', 'rcs', 'z', 'range_m'])
    parser.add_argument('--x_range', type=float, nargs=2, default=[-10.0, 210.0],
                        metavar=('XMIN', 'XMAX'),
                        help='BEV x 固定范围,防序列动画视角跳变(默认 -10 210,ROI 外留边)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-25.0, 25.0],
                        metavar=('YMIN', 'YMAX'),
                        help='BEV y 固定范围(默认 -25 25,ROI 外留边)')
    parser.add_argument('--cbar_range', type=float, default=16.0,
                        help='点云热图固定对称范围 ±该值(m/s,默认 16=抽样 p99);0=逐帧自适应')
    args = parser.parse_args()

    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.msr.msr_dataset import MsrDataset
    from visualize_msr import plot_frame

    save, overlap = args.save, args.overlap
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path('output/res_viz_seq') / Path(args.cfg_file).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读帧 dataset(直读 POINTS/LABELS/PARAMS,不依赖 infos)----
    mcfg = cfg_from_yaml_file(Path(args.cfg_file), EasyDict())
    base = MsrDataset(dataset_cfg=mcfg.DATA_CONFIG, class_names=list(mcfg.CLASS_NAMES),
                      training=False, root_path=None)

    # ---- 模型 + tracker 式预处理模板(内存点云 → prepare_data,无 dataloader)----
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    net = build_network(mcfg.MODEL, num_class=len(mcfg.CLASS_NAMES), dataset=base)
    state = torch.load(str(args.ckpt), map_location='cpu')
    net.load_state_dict(state['model_state'], strict=True)
    net.cuda().eval()
    print('[viz] ckpt loaded: %s (epoch %s)' % (args.ckpt, state.get('epoch', '?')))
    tmpl = DatasetTemplate(dataset_cfg=mcfg.DATA_CONFIG, class_names=list(mcfg.CLASS_NAMES),
                           training=False, root_path=Path(mcfg.DATA_CONFIG.DATA_PATH))

    # ---- 连续帧号区间 + testing 帧防护(同 visualize_msr.pick_frames 口径)----
    start, width = int(args.start_sid), len(args.start_sid)
    sids = ['%0*d' % (width, i) for i in range(start, start + args.num * args.stride, args.stride)]
    testing, valset = set(), set()
    for name, bucket in [('testing.txt', testing), ('val.txt', valset)]:
        f = base.root_path / 'IMAGESETS' / name
        if f.exists():
            bucket.update(x.strip() for x in open(f).readlines())

    def in_tv(sid):
        """
        split 归属判断: 该帧是否属 val/testing(用于黑色外框标记)
        """
        return sid in testing or sid in valset

    print('=== visualize_msr_seq: save=%s overlap=%d | %d frames from %s (stride %d), model=%s ==='
          % (save, overlap, len(sids), args.start_sid, args.stride, Path(args.cfg_file).stem))
    n_done = 0
    for sid in sids:
        if sid in testing and not args.allow_testing:
            print('  %s -> skip (testing split; --allow_testing 放行)' % sid)
            continue
        if not (base.root_path / 'POINTS' / ('%s.bin' % sid)).exists():
            print('  %s -> skip (POINTS missing)' % sid)
            continue

        # ---- 检测: 内存点云直推 ----
        pts = base.get_radar(sid)                     # (N,18) MSR_FEATURE_ORDER 全列
        pred, infer_ms = None, 0.0
        if pts.shape[0] > 0:
            data_dict = tmpl.prepare_data({'points': pts, 'frame_id': sid})
            batch = tmpl.collate_batch([data_dict])
            load_data_to_gpu(batch)
            t0 = time.time()
            with torch.no_grad():
                pred_dicts, _ = net(batch)
            infer_ms = (time.time() - t0) * 1e3
            pred = {k: v.detach().cpu().numpy() for k, v in pred_dicts[0].items()
                    if hasattr(v, 'numpy')}
        elif pts.shape[0] == 0:
            print('  %s -> 0 points, inference skipped' % sid)

        # ---- 落盘门控: save!=none 出图;overlap=0 时已存在 PNG 跳过 ----
        out, n_pts, n_gt, n_p, cls = None, pts.shape[0], -1, -1, []
        do_plot = (save != 'none')
        if do_plot and overlap == 0 and (out_dir / ('seq_bev_%s.png' % sid)).exists():
            do_plot = False
        if do_plot:
            out, n_pts, n_gt, n_p, cls = plot_frame(
                base, sid, 'seq', out_dir, args.color_by, pred=pred,
                x_range=args.x_range, y_range=args.y_range,
                cbar_range=args.cbar_range or None,
                border_black=in_tv(sid))
        n_done += 1
        print('  %s -> %s  (pts=%d gt=%s pred=%s infer=%.1fms classes=%s)'
              % (sid, out.name if out else '(not saved)', n_pts,
                 n_gt if n_gt >= 0 else '-', n_p if n_p >= 0 else '-',
                 infer_ms, cls))
    print('=== done, %d frames processed, output dir: %s ===' % (n_done, out_dir))
    if save in ('gif', 'mp4'):
        save_animation(out_dir, save)


if __name__ == '__main__':
    main()
