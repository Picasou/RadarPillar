from schemas import Cfg, VDS, FRAME
import loader
import preprocessor
import detector
import filter
import matcher
import manager
import evaluator


class Tracker:
    def __init__(self, cfg_path: str) -> None:
        self.cfg = Cfg.get_cfg(cfg_path)
        self.cfg.isvalid()
        self.trks = []

        self.loader       = loader.Loader(self.cfg)
        self.preprocessor = preprocessor.Preprocessor(self.cfg)
        self.detector     = detector.Detector(self.cfg)
        self.filter       = filter.KalmanFilter(self.cfg)
        self.matcher      = matcher.Matcher(self.cfg)
        self.manager      = manager.TrackManager(self.cfg)
        self.evaluator    = evaluator.Evaluator(self.cfg)

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
            for frame in frames:
                self.step(frame, vds)

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

    def step(self, frame: FRAME, vds: VDS) -> None:
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