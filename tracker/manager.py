"""轨迹管理: 跟踪器集合的创建/更新/删除."""
import configparser


class TrackerManager:
    """
    in : Matches, objs (当前检测), FRAME
    out: 原地维护 self.trackers (创建/更新/删除活跃轨迹)
    """

    def __init__(self, cfg: configparser.ConfigParser):
        self.match_thresh = cfg.getfloat('tracker', 'match_thresh', fallback=3.0)
        self.max_age = cfg.getint('tracker', 'max_age', fallback=3)
        self.min_hits = cfg.getint('tracker', 'min_hits', fallback=3)
        self.dt = cfg.getfloat('tracker', 'dt', fallback=0.1)
        self.trackers = []   # 活跃轨迹列表
        self.frame_count = 0
