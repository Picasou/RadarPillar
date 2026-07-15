"""性能评估: MOT 指标 (MOTA/ID切换/跟踪碎片率)."""
from pathlib import Path


class Evaluator:
    """
    in : online→FRAME / evaluate→history[(gts, tracks_list)]
    out: MOT 指标 (MOTA / ID 切换 / 碎片率) 报告或可视化
    """

    def __init__(self, gt_dir='data/tracker_gt', trk_dir='output/tracks'):
        self.gt_dir = Path(gt_dir)
        self.trk_dir = Path(trk_dir)
