from __future__ import annotations

from pathlib import Path

from .schemas import Cfg, VDS, FRAME, FRAMEs, Trk
from .utils.common import (load_data_cfg, c_points_prepare)
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
        self.trks: list[Trk] = []
        self.accum_frames = self.cfg.RUN.accum_frames
        self.point_cloud_range = load_data_cfg().POINT_CLOUD_RANGE

        self.loader    = loader.Loader(self.cfg)
        self.detector  = detector.Detector(self.cfg)
        self.updater   = updater.Updater(self.cfg)
        self.matcher   = matcher.Matcher(self.cfg)
        self.manager   = manager.TrackerManager(self.cfg)
        self.evaluator = evaluator.Evaluator(self.cfg)
        self.visualizer = visualizer.Visualizer(self.cfg, class_names=self.detector.class_names)

    def run(self) -> None:
        run_mode  = self.cfg.RUN.mode       # 0=display  1=normal  2=regress
        eval_mode = self.cfg.EVALUATE.type  # 0=off  1=online  2=offline
        is_visualize = (self.cfg.VISUAL.enable == 1)

        history = []
        for path in self.cfg.DATA.paths:
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)
            if is_visualize:
                self.visualizer.begin_seq(Path(path).name)

            tracks_list = []
            for i, frame in enumerate(frames.Lst):

                self.step(frame, frames, self.trks, vds, i)

                if run_mode != 0:
                    tracks_list.append([t.copy() for t in self.trks if t.obstacle_prob])
                    if eval_mode == 1:
                        self.evaluator.online(frame)

                if run_mode == 2 and self.cfg.RUN.overlap == 1:
                    self.write(frame)

            if run_mode != 0:
                history.append((frame.gts, tracks_list.copy()))

            if is_visualize:
                self.visualizer.on_seq_end()

        if eval_mode == 2:
            self.evaluator.evaluate(history)

    def step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS, i: int) -> None:
        # 1. 点云准备
        frame.proc.points = c_points_prepare(frames, i, vds, self.accum_frames, self.point_cloud_range)
        frame.frame_id = str(frame.pts.Lst[0].frame) if frame.pts.Lst else ''

        # 2. 检测
        objs = self.detector.run(frame)

        # 3. 预测
        self.updater.predict(trks, frame.vdd, vds.cycle_s)

        # 4. 关联
        matches = self.matcher.run(trks, objs)

        # 5. 更新
        self.updater.run(matches, vds.cycle_s)
        
        # 6. 航迹管理
        self.manager.run(matches, trks, vds.cycle_s)

        # 7. 可视化
        if self.cfg.VISUAL.enable == 1:
            self.visualizer.run(frame, trks)
