from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .schemas import Cfg, VDS, FRAME, FRAMEs, Trk
from .utils.common import (load_data_cfg, c_points_prepare)
from .utils.rw_struct import struct_write, Raw_TrkHead, Raw_Trk
from . import loader
from . import detector
from . import matcher
from . import updater
from . import manager
from . import evaluator
from . import visualizer


class Tracker:
    """
    全链路编排
    """
    def __init__(self, cfg_path: str) -> None:
        # 参数初始化
        self.cfg = Cfg.get_cfg(cfg_path)
        self.cfg.isvalid()
        self.mode = self.cfg.RUN.mode   # 0=display 1=just_model 2=full
        self.do_save = (self.cfg.RUN.save == 1)
        self.eval_mode = self.cfg.EVALUATE.type   # 0=off 1=online 2=offline
        if self.eval_mode and self.cfg.RUN.mode != 2:
            print('[tracker] warning: EVALUATE.type=%d 仅在 RUN.mode=2 下生效, 本次评估关闭'% self.eval_mode)
        self.is_visualize = (self.cfg.VISUAL.enable == 1)
        self.trks: list[Trk] = []
        self.accum_frames = self.cfg.RUN.accum_frames

        import yaml as _yaml
        with open(self.cfg.MODEL.cfg, 'r', encoding='utf-8') as f:
            _mcfg = _yaml.safe_load(f) or {}
        self.point_cloud_range = (
            (_mcfg.get('DATA_CONFIG') or {}).get('POINT_CLOUD_RANGE')
            or load_data_cfg().POINT_CLOUD_RANGE)  # type: ignore[attr-defined]

        # 模块初始化
        self.loader    = loader.Loader(self.cfg)
        self.detector  = None if self.mode == 0 else detector.Detector(self.cfg)
        self.updater   = updater.Updater(self.cfg)
        self.matcher   = matcher.Matcher(self.cfg)
        self.manager   = manager.TrackerManager(self.cfg)
        self.evaluator = evaluator.Evaluator(
            self.cfg, class_names=self.detector.class_names if self.detector else None)
        self.visualizer = visualizer.Visualizer(
            self.cfg, class_names=self.detector.class_names if self.detector else None)

    def run(self) -> None:
        history = []
        for path in self.cfg.DATA.paths:
            self.trks = []          # 序列边界重置: 航迹不跨序列 (P0-1)
            self.updater.reset()    # 重置: 类型后验 / IMM bank
            trk_rows = []           # 序列级结果收集: [(frame_id, [Trk|Obj...])], 序列末统一写 bin
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)
            if not frames.Lst:      
                continue

            if self.eval_mode == 1 and self.mode == 2:
                self.evaluator.on_seq_start(path)

            if self.is_visualize:
                aix_lim = (min(p.x_m for f in frames.Lst for p in f.pts.Lst),
                       max(p.x_m for f in frames.Lst for p in f.pts.Lst),
                       min(p.y_m for f in frames.Lst for p in f.pts.Lst),
                       max(p.y_m for f in frames.Lst for p in f.pts.Lst)) \
                    if any(f.pts.Lst for f in frames.Lst) else None
                self.visualizer.begin_seq(Path(path).name, path, data_extent=aix_lim,
                                          is_test=Path(path).name in (self.cfg.VISUAL.test_val or []))

            seq_history = []        # 逐帧 (gts, 输出航迹快照), 评估配对用 (P0-2)
            for i, frame in enumerate(frames.Lst):

                objs = self.tracker_step(frame, frames, self.trks, vds, i)

                if self.mode == 2:
                    live = [t for t in self.trks if t.obstacle_prob]
                    if self.eval_mode == 1:
                        self.evaluator.online(frame, [evaluator.snap_trk(t) for t in live])
                    elif self.eval_mode == 2:
                        seq_history.append((frame.gts, [evaluator.snap_trk(t) for t in live]))
                    if self.do_save:
                        trk_rows.append((frame.frame_id, [copy.deepcopy(t) for t in live]))
                elif self.do_save and self.mode == 1:
                    trk_rows.append((frame.frame_id, [copy.deepcopy(t) for t in objs]))

            if self.mode == 2:
                history.append((path, seq_history))
            if self.do_save and trk_rows:
                self.write(path, trk_rows)

            if self.is_visualize:
                self.visualizer.on_seq_end()
            if self.eval_mode == 1 and self.mode == 2: 
                self.evaluator.on_seq_end(path)

        if self.eval_mode == 1 and self.mode == 2:
            self.evaluator.on_dataset_end()
        if self.eval_mode == 2 and self.mode == 2:
            self.evaluator.evaluate(history)

    def tracker_step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS, i: int) -> list:
        # 1. 加载数据
        frame.proc.points = c_points_prepare(frames, i, vds, self.accum_frames, self.point_cloud_range)
        frame.frame_id = '%06d' % i      # bin 点级 frame 号未填(恒 0), 用帧序号

        # 2. 检测
        objs = []
        if self.mode >= 1 and self.detector is not None:
            objs = self.detector.run(frame)

        if self.mode == 2:
            # 3. 预测
            self.updater.predict(trks, frame.vdd, vds.cycle_s)
            # 4. 关联
            matches = self.matcher.run(trks, objs)
            # 5. 更新
            self.updater.run(matches, vds.cycle_s)
            # 6. 管理
            self.manager.run(matches, trks, vds.cycle_s)

        # 7. 可视化
        if self.is_visualize:
            self.visualizer.run(frame, objs, trks if self.mode == 2 else [])

        return objs

    def write(self, seq_path: str, trk_rows: list) -> None:
        out_dir = Path(seq_path) / self.loader.relpath
        suffix = '00000' if self.cfg.RUN.overlap == 1 else '00001'
        rec_file = out_dir / ('0201.%s.bin' % suffix)
        head_file = out_dir / ('0200.%s.bin' % suffix)
        if rec_file.exists() and self.cfg.RUN.overlap == 0:
            return
        heads, records = [], []
        for frame_id, items in trk_rows:
            h = Raw_TrkHead()
            h.version, h.frame_cnt, h.trk_num, h.reserved = 1, int(frame_id), len(items), 0
            heads.append(h)
            for t in items:
                r = Raw_Trk()
                if isinstance(t, Trk):
                    # 量化口径对齐数据源实测: 位置/速度/加速度/朝向/尺寸 ×100, std ×10000, 概率/状态直存
                    r.id = int(t.id)
                    r.x_m, r.y_m, r.z_m = (int(round(v * 100)) for v in (t.x_m, t.y_m, t.z_m))
                    r.vx_mps, r.vy_mps = (int(round(v * 100)) for v in (t.vx_mps, t.vy_mps))
                    r.ax_mps2, r.ay_mps2 = (int(round(v * 100)) for v in (t.ax_mps2, t.ay_mps2))
                    r.heading_deg = int(round(t.heading_deg * 100))
                    r.width_m, r.length_m, r.height_m = (int(round(v * 100)) for v in
                                                         (t.width_m, t.length_m, t.height_m))
                    r.type = int(t.type)
                    r.type_confi = int(t.type_confi)
                    r.lifetime_s = int(round(t.lifetime_s * 100))
                    r.motion_status = int(t.motion_status)
                    r.measurement_status = int(t.measurement_status)
                    r.existence_prob = int(t.existence_prob)
                    r.obstacle_prob = int(t.obstacle_prob)
                    r.passable_status = int(t.passable_status)
                    # 本链路经 ego 运动补偿, 输出为绝对量 (0=absolute)
                    r.rel_vel, r.rel_acc = 0, 0
                    r.vx_std_mps, r.vy_std_mps = (max(0, min(65535, int(round(v * 10000))))
                                                  for v in (t.vx_std_mps, t.vy_std_mps))
                    r.xy_vel_cov = int(round(t.xy_vel_cov * 10000)) & 0xFFFF
                    r.ax_std_mps2, r.ay_std_mps2 = (max(0, min(65535, int(round(v * 10000))))
                                                    for v in (t.ax_std_mps2, t.ay_std_mps2))
                    r.xy_acc_cov = int(round(t.xy_acc_cov * 10000)) & 0xFFFF
                    r.x_std_m, r.y_std_m, r.z_std_m = (max(0, min(65535, int(round(v * 10000))))
                                                       for v in (t.x_std_m, t.y_std_m, t.z_std_m))
                    r.xy_pos_cov = int(round(t.xy_pos_cov * 10000)) & 0xFFFF
                    r.heading_std = int(round(t.heading_std_deg))
                    r.yaw_rate_degs = int(round(t.yaw_rate_degs * 100))
                    r.yaw_rate_std = int(round(t.yaw_rate_std_degs))
                    r.length_std, r.width_std, r.height_std = (int(round(v)) for v in
                                                               (t.length_std_m, t.width_std_m,
                                                                t.height_std_m))
                else:   # Obj(检测, mode=1): 非航迹 id 恒 0, 航迹级字段无来源置 0
                    r.id = 0
                    r.x_m, r.y_m = int(round(t.x * 100)), int(round(t.y * 100))
                    r.z_m = 0
                    r.vx_mps, r.vy_mps = int(round(t.vx * 100)), int(round(t.vy * 100))
                    r.ax_mps2, r.ay_mps2 = 0, 0
                    r.heading_deg = int(round(t.heading * 100))
                    r.width_m, r.length_m = int(round(t.width * 100)), int(round(t.length * 100))
                    r.height_m = 0
                    r.type = int(t.type)
                    r.type_confi = int(round(t.score * 100))
                records.append(r)
        struct_write(str(rec_file), records, heads=heads, head_filepath=str(head_file))
        print('  [tracker] %d frames, %d trks -> %s' % (len(heads), len(records), rec_file))


def main():
    """
    模块入口: 解析 --cfg 并实例化/运行 Tracker (需 python -m tracker.tracker 方式启动)
    """
    parser = argparse.ArgumentParser(description='run tracker full chain on seq data')
    parser.add_argument('--cfg', type=str,  default=str(Path(__file__).parent / 'cfg' / 'cfg.yaml'))
    args = parser.parse_args()

    trk = Tracker(args.cfg)
    trk.run()


if __name__ == '__main__':
    main()
