from __future__ import annotations
import numpy as np

from schemas import Cfg, VDS, FRAME, FRAMEs, Trk
from loader import Loader
from utils.common import (load_data_cfg, accumulate_points, crop_range, compensate_trks)
import detector
import filter
import matcher
import manager
import evaluator


class Tracker:
    def __init__(self, cfg_path: str) -> None:
        self.cfg = Cfg.get_cfg(cfg_path)
        self.cfg.isvalid()
        self.trks: list[Trk] = []
        self.accum_frames = self.cfg.RUN.accum_frames
        self.point_cloud_range = load_data_cfg().POINT_CLOUD_RANGE

        self.loader    = Loader(self.cfg)
        self.detector  = detector.Detector(self.cfg)
        self.filter    = filter.KalmanFilter(self.cfg)
        self.matcher   = matcher.Matcher(self.cfg)
        self.manager   = manager.TrackManager(self.cfg)
        self.evaluator = evaluator.Evaluator(self.cfg)

    def run(self) -> None:
        mode = self.cfg.RUN.mode
        is_display = (mode == 0)
        is_regress = (mode == 2)
        is_evaluate_online = (self.cfg.EVALUATE.type == 1)
        is_evaluate_offline = (self.cfg.EVALUATE.type == 2)
        is_visualize = (self.cfg.VISUALIZE.enable == 1)

        history = []
        for path in self.cfg.DATA.paths:
            frames = self.loader.getframes(path)
            vds    = self.loader.getvds(path)

            tracks_list = []
            for i, frame in enumerate(frames.Lst):
                start = max(0, i - self.accum_frames + 1)
                window = FRAMEs(num=i - start + 1, Lst=frames.Lst[start:i + 1])
                self.step(frame, window, self.trks, vds)

                if not is_display:
                    tracks_list.append([t.copy() for t in self.trks])
                    if is_evaluate_online:
                        self.evaluator.online(frame)

                if is_visualize:
                    self.evaluator.visualize(frame)

                if is_regress and self.cfg.RUN.overlap == 1:
                    self.write(frame)

            if not is_display:
                history.append((frame.gts, tracks_list.copy()))

        if is_evaluate_offline:
            self.evaluator.evaluate(history)

    def step(self, frame: FRAME, frames: FRAMEs, trks: list[Trk], vds: VDS) -> None:
        # 1. 点云准备
        points = accumulate_points(frames, vds, self.accum_frames)
        if points.shape[0] > 0:
            points = crop_range(points, self.point_cloud_range)
        frame.points   = points.astype(np.float32, copy=False)
        frame.frame_id = str(frame.pts.Lst[0].frame) if frame.pts.Lst else ''

        # 2. trk 状态补偿
        compensate_trks(trks, frame.vdd, vds.cycle_s)

        # 3. 检测
        objs = self.detector.run(frame)
        # 4. 预测
        self.filter.predict(trks)
        # 5. 关联
        matches = self.matcher.run(trks, objs)
        # 6. 更新
        self.filter.update(matches)
        # 7. 航迹管理
        self.manager.run(matches, objs, frame)


if __name__ == '__main__':
    cfg_path = r"cfg\cfg.yaml"
    Tracker(cfg_path).run()
