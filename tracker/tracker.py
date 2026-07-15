from __future__ import annotations

from .schemas import Cfg, VDS, FRAME, FRAMEs, Trk
from .loader import Loader
from .utils.common import (load_data_cfg, prepare_points)
from . import detector
from . import filter
from . import matcher
from . import manager
from . import evaluator


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

        self.loader    = Loader(self.cfg)
        self.detector  = detector.Detector(self.cfg)
        self.filter    = filter.Filter(self.cfg)
        self.matcher   = matcher.Matcher(self.cfg)
        self.manager   = manager.TrackManager(self.cfg)
        self.evaluator = evaluator.Evaluator(self.cfg)

    def run(self) -> None:
        run_mode  = self.cfg.RUN.mode       # 0=display  1=normal  2=regress
        eval_mode = self.cfg.EVALUATE.type  # 0=off  1=online  2=offline
        is_visualize = (self.cfg.VISUALIZE.enable == 1)

        history = []
        for path in self.cfg.DATA.paths:
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)

            tracks_list = []
            for i, frame in enumerate(frames.Lst):
                
                self.step(frame, frames, self.trks, vds, i)

                if run_mode != 0:
                    tracks_list.append([t.copy() for t in self.trks])
                    if eval_mode == 1:
                        self.evaluator.online(frame)

                if is_visualize:
                    self.evaluator.visualize(frame)

                if run_mode == 2 and self.cfg.RUN.overlap == 1:
                    self.write(frame)

            if run_mode != 0:
                history.append((frame.gts, tracks_list.copy()))

        if eval_mode == 2:
            self.evaluator.evaluate(history)

    def step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS, i: int) -> None:
        # 1. 点云准备
        frame.proc.points = prepare_points(frames, i, vds, self.accum_frames, self.point_cloud_range)
        frame.frame_id = str(frame.pts.Lst[0].frame) if frame.pts.Lst else ''
        # 2. 检测
        objs = self.detector.run(frame)
        # 3. 预测
        self.filter.predict(trks, frame.vdd, vds.cycle_s)
        # 4. 关联
        matches = self.matcher.run(trks, objs)
        # 5. 更新
        self.filter.update(matches)
        # 6. 航迹管理
        self.manager.run(matches, objs, frame)
