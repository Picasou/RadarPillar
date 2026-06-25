"""卡尔曼滤波: 对每个轨迹做常速度运动估计."""
import numpy as np


class KalmanBoxTracker:
    """单目标卡尔曼跟踪器, 状态=[x,y,vx,vy], 观测=[x,y]."""

    count = 0  # 全局轨迹 ID 计数

    def __init__(self, det_center: np.ndarray, process_var=1.0, measure_var=0.5):
        self.x = np.array([det_center[0], det_center[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 10.0
        self.Q = np.diag([process_var] * 2 + [process_var * 2] * 2)  # 过程噪声
        self.R = np.eye(2) * measure_var                              # 量测噪声
        self.H = np.zeros((2, 4)); self.H[0, 0] = self.H[1, 1] = 1.0  # 取 x,y

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history: list[np.ndarray] = []
