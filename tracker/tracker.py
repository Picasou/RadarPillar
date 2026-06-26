import argparse
import yaml
from schemas import (Cfg,
    VDS, VDD, PT, PTs, GT, GTs, FRAME, FRAMEs,
    Objs, Trks, Matches,
    CfgRun, CfgVds, CfgData, CfgModel,
    CfgFilter, CfgFilterPara, CfgFilterParaKf,
    CfgMatch, CfgVisualize, CfgEvaluate, CfgManager)

import loader       as loader
import preprocessor as preer
import detector     as detector
import filter       as filter
import matcher      as matcher
import manager      as manager
import evaluator    as evaluator


class Tracker:
    def __init__(self, cfg_path: str) -> None:
        self.cfg = Cfg.get_cfg(cfg_path)
        self.trks = []  # 轨迹列表
        
        self.cfg.isvalid()  # 校验配置，失败抛异常

        # 各模块初始化
        self.loader       = loader.Loader(self.cfg)
        self.preprocessor = preer.Preprocessor(self.cfg)
        self.detector     = detector.Detector(self.cfg)
        self.filter       = filter.KalmanFilter(self.cfg)
        self.matcher      = matcher.Matcher(self.cfg)
        self.manager      = manager.TrackManager(self.cfg)
        self.evaluator    = evaluator.Evaluator(self.cfg)

    def run(self) -> None:
        mode = self.cfg.RUN.mode
        is_display = (mode == 0)
        is_regress = (mode == 2)

        history = []
        for path in self.cfg.DATA.paths:
            frames = self.loader.getframes(path)
            vds = self.loader.getvds(path)
            tracks_list = []

            for frame in frames:
                self.step(frame, vds)

                if not is_display:
                    tracks_list.append([t.copy() for t in self.trks])
                    if self.cfg.EVALUATE.type == 1:   
                        self.evaluator.online(frame)

                if self.cfg.VISUALIZE.enable == 1: 
                    self.evaluator.visualize(frame)

                if is_regress and self.cfg.RUN.overlap == 1: 
                    self.write(frame)

            if not is_display:
                history.append((frame.gts, tracks_list.copy()))

        if self.cfg.EVALUATE.type == 2:
            self.evaluator.evaluate(history)

    def step(self, frame, vds):
        # 1 预处理
        frame = self.preprocessor.run(frame, vds)
        # 2 检测
        objs = self.detector.run(frame)
        # 3 预测
        self.filter.predict(self.trks)
        # 4 关联
        matches = self.matcher.run(self.trks, objs)
        # 5 更新
        self.filter.update(matches)
        # 6 航迹管理 
        self.manager.run(matches, objs, frame)


if __name__ == '__main__':
    cfg_path = r"cfg\cfg.yaml"
    Tracker(cfg_path).run()
