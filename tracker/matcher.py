"""数据关联: 匈牙利算法 + 距离门限."""


class Matcher:
    """预测轨迹与当前检测的关联."""

    def __init__(self, thresh: float = 3.0):
        self.thresh = thresh                # 关联距离门限 (m)
