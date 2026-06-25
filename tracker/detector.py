"""检测入口: 调用 RadarPillar 模型推理, 输出 3D 框."""
import configparser


class Detector:
    """封装 RadarPillar 推理."""

    def __init__(self, cfg: configparser.ConfigParser):
        self.ckpt = cfg['detector']['ckpt']            # 权重路径
        self.cfg_path = cfg['detector']['cfg']          # 模型配置
        self.score_thresh = cfg.getfloat('detector', 'score_thresh', fallback=0.3)
        self.model = None                               # 推理模型 (加载后赋值)
