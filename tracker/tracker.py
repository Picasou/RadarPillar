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
        self.trks: list[Trk] = []
        self.accum_frames = self.cfg.RUN.accum_frames
        # 点云过滤范围以模型训练口径为准(MODEL.cfg 的 POINT_CLOUD_RANGE),
        # 兜底旧 astyx data cfg(点云截短会让模型看不到远处目标)
        import yaml as _yaml
        with open(self.cfg.MODEL.cfg, 'r', encoding='utf-8') as f:
            _mcfg = _yaml.safe_load(f) or {}
        self.point_cloud_range = (
            (_mcfg.get('DATA_CONFIG') or {}).get('POINT_CLOUD_RANGE')
            or load_data_cfg().POINT_CLOUD_RANGE)

        self.loader    = loader.Loader(self.cfg)
        # display(0) 不加载模型: 只读数据可视化, detector 惰性不建
        self.detector  = None if self.mode == 0 else detector.Detector(self.cfg)
        self.updater   = updater.Updater(self.cfg)
        self.matcher   = matcher.Matcher(self.cfg)
        self.manager   = manager.TrackerManager(self.cfg)
        self.evaluator = evaluator.Evaluator(self.cfg)
        self.visualizer = visualizer.Visualizer(
            self.cfg, class_names=self.detector.class_names if self.detector else None)

    def run(self) -> None:
        run_mode  = self.mode                   # 0=display  1=just_model  2=full
        do_save   = (self.cfg.RUN.save == 1)
        eval_mode = self.cfg.EVALUATE.type  # 0=off  1=online  2=offline
        is_visualize = (self.cfg.VISUAL.enable == 1)

        history = []
        for path in self.cfg.DATA.paths:
            trk_rows = []           # 序列级航迹收集: [(frame_id, [Trk...])], 序列末统一写 bin
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)
            if is_visualize:
                # 全程点云外沿 → 固定坐标范围(外沿+3m, 跨帧一致)
                ext = (min(p.x_m for f in frames.Lst for p in f.pts.Lst),
                       max(p.x_m for f in frames.Lst for p in f.pts.Lst),
                       min(p.y_m for f in frames.Lst for p in f.pts.Lst),
                       max(p.y_m for f in frames.Lst for p in f.pts.Lst)) \
                    if any(f.pts.Lst for f in frames.Lst) else None
                self.visualizer.begin_seq(Path(path).name, path, data_extent=ext,
                                          is_test=Path(path).name in (self.cfg.VISUAL.test_val or []))

            tracks_list = []
            for i, frame in enumerate(frames.Lst):

                self.step(frame, frames, self.trks, vds, i)

                if run_mode == 2:
                    tracks_list.append([copy.deepcopy(t) for t in self.trks if t.obstacle_prob])
                    if eval_mode == 1:
                        self.evaluator.online(frame)
                    if do_save:
                        trk_rows.append((frame.frame_id,
                                         [copy.deepcopy(t) for t in self.trks if t.obstacle_prob]))

            if run_mode == 2:
                history.append((frame.gts, tracks_list.copy()))
                if do_save and trk_rows:
                    self.write(Path(path).name, trk_rows)

            if is_visualize:
                self.visualizer.on_seq_end()

        if eval_mode == 2:
            self.evaluator.evaluate(history)

    def step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS, i: int) -> None:
        # 1. 点云准备(三模式共用: display 也要点云出图)
        frame.proc.points = c_points_prepare(frames, i, vds, self.accum_frames, self.point_cloud_range)
        frame.frame_id = '%06d' % i      # bin 点级 frame 号未填(恒 0), 用帧序号

        # 2. 检测(mode>=1; mode=0 display 零处理)
        objs = []
        if self.mode >= 1:
            objs = self.detector.run(frame)

        # 3~6. 航迹链路(仅 mode=2: 预测→关联→更新→管理)
        if self.mode == 2:
            self.updater.predict(trks, frame.vdd, vds.cycle_s)
            matches = self.matcher.run(trks, objs)
            self.updater.run(matches, vds.cycle_s)
            self.manager.run(matches, trks, vds.cycle_s)

        # 7. 可视化(0=点云+GT / 1=+det / 2=+trk 带 ID)
        if self.cfg.VISUAL.enable == 1:
            self.visualizer.run(frame, objs, trks if self.mode == 2 else [])

    def write(self, seq_name: str, trk_rows: list) -> None:
        """
        航迹落盘: 序列上桌航迹按 bin 格式写 0200(帧头)/0201(目标) 仿数据源布局,
        overlap=0 且文件已存在时跳过
        """
        out_dir = Path('output/tracks') / seq_name
        rec_file = out_dir / '0201.00000.bin'
        head_file = out_dir / '0200.00000.bin'
        if rec_file.exists() and self.cfg.RUN.overlap == 0:
            return
        heads, records = [], []
        for frame_id, trks in trk_rows:
            h = Raw_TrkHead()
            h.version, h.frame_cnt, h.trk_num, h.reserved = 1, int(frame_id), len(trks), 0
            heads.append(h)
            for t in trks:
                r = Raw_Trk()
                r.id = int(t.id)
                r.x, r.y, r.z = (int(round(v * 100)) for v in (t.x_m, t.y_m, t.z_m))
                r.vx, r.vy = (int(round(v * 100)) for v in (t.vx_mps, t.vy_mps))
                r.ax, r.ay = (int(round(v * 100)) for v in (t.ax_mps2, t.ay_mps2))
                r.heading = int(round(t.heading_deg * 100))
                r.width, r.length, r.height = (int(round(v * 100)) for v in
                                               (t.width_m, t.length_m, t.height_m))
                r.classification = int(t.type)
                r.confidence = int(t.existence_prob)
                records.append(r)
        struct_write(str(rec_file), records, heads=heads, head_filepath=str(head_file))
        print('  [tracker] %d frames, %d trks -> %s' % (len(heads), len(records), rec_file))
