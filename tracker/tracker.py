from __future__ import annotations

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
        self.cfg = Cfg.get_cfg(cfg_path)
        self.cfg.isvalid()
        self.mode = self.cfg.RUN.mode   # 0=display 1=just_model 2=full
        self.do_save = (self.cfg.RUN.save == 1)
        self.eval_mode = self.cfg.EVALUATE.type   # 0=off 1=online 2=offline
        self.is_visualize = (self.cfg.VISUAL.enable == 1)
        self.trks: list[Trk] = []
        self.accum_frames = self.cfg.RUN.accum_frames

        # 按照模型配置获得 point_cloud_range
        import yaml as _yaml
        with open(self.cfg.MODEL.cfg, 'r', encoding='utf-8') as f:
            _mcfg = _yaml.safe_load(f) or {}
        self.point_cloud_range = (
            (_mcfg.get('DATA_CONFIG') or {}).get('POINT_CLOUD_RANGE')
            or load_data_cfg().POINT_CLOUD_RANGE)

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
            self.updater.reset()    #           重置: 类型后验 / IMM bank
            trk_rows = []           # 序列级结果收集: [(frame_id, [Trk|Obj...])], 序列末统一写 bin
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)
            if not frames.Lst:      # 空序列守卫: 后续不再引用未绑定 frame
                print('  [tracker] %s: 0 frames, skip' % Path(path).name)
                continue
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

                out_trks = []
                if self.mode == 2:
                    out_trks = [copy.deepcopy(t) for t in self.trks if t.obstacle_prob]
                    seq_history.append((frame.gts, out_trks))
                    if self.eval_mode == 1:     # online: 逐帧记账
                        self.evaluator.online(frame, out_trks)
                if self.do_save and self.mode >= 1:
                    # mode=2 落航迹 / mode=1 落检测(非航迹, 落盘 id 恒 0)
                    rows = out_trks if self.mode == 2 else objs
                    trk_rows.append((frame.frame_id, [copy.deepcopy(t) for t in rows]))

            if self.mode == 2:
                history.append(seq_history)
            if self.do_save and trk_rows:
                self.write(path, trk_rows)

            if self.is_visualize:
                self.visualizer.on_seq_end()
            if self.eval_mode == 1 and self.mode == 2:   # online 模式才逐序列打印 (offline 由 evaluate 汇总)
                self.evaluator.on_seq_end(Path(path).name)

        if self.eval_mode == 2 and self.mode == 2:
            self.evaluator.evaluate(history)

    def tracker_step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS, i: int) -> list:
        # 1. 加载数据
        frame.proc.points = c_points_prepare(frames, i, vds, self.accum_frames, self.point_cloud_range)
        frame.frame_id = '%06d' % i      # bin 点级 frame 号未填(恒 0), 用帧序号

        # 2. 检测
        objs = []
        if self.mode >= 1:
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
                    r.id = int(t.id)
                    r.x, r.y, r.z = (int(round(v * 100)) for v in (t.x_m, t.y_m, t.z_m))
                    r.vx, r.vy = (int(round(v * 100)) for v in (t.vx_mps, t.vy_mps))
                    r.ax, r.ay = (int(round(v * 100)) for v in (t.ax_mps2, t.ay_mps2))
                    r.heading = int(round(t.heading_deg * 100))
                    r.width, r.length, r.height = (int(round(v * 100)) for v in
                                                   (t.width_m, t.length_m, t.height_m))
                    r.confidence = int(t.existence_prob)
                else:   # Obj(检测, mode=1): 非航迹 id 恒 0, z/ax/ay/height 无来源置 0
                    r.id = 0
                    r.x, r.y = int(round(t.x * 100)), int(round(t.y * 100))
                    r.z = 0
                    r.vx, r.vy = int(round(t.vx * 100)), int(round(t.vy * 100))
                    r.ax, r.ay = 0, 0
                    r.heading = int(round(t.heading * 100))
                    r.width, r.length = int(round(t.width * 100)), int(round(t.length * 100))
                    r.height = 0
                    r.confidence = int(round(t.score * 100))
                r.classification = int(t.type)
                records.append(r)
        struct_write(str(rec_file), records, heads=heads, head_filepath=str(head_file))
        print('  [tracker] %d frames, %d trks -> %s' % (len(heads), len(records), rec_file))
