"""RPiN 阶段5 E3：VDC 多帧速度运动补偿（radar motion compensation）。
纯函数 `compensate_motion`，对 (N, 7) 原始点云按速度反推时间偏移做空间对齐。
模块级常量 RADAR_FEATURE_ORDER 作为权威定义（VodDataset 导入复用）。
"""
import numpy as np

# 权威特征列序（与 VodDataset.radar_feature_order 一致；VodDataset 通过 `from .vdc import RADAR_FEATURE_ORDER` 导入）
RADAR_FEATURE_ORDER = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
# 索引常量
_X, _Y, _Z, _RCS, _VR, _VRC, _TIME = (RADAR_FEATURE_ORDER.index(n) for n in ('x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'))


def compensate_motion(points: np.ndarray, cfg: dict = None) -> np.ndarray:
    """对 (N, 7) 原始点云做速度运动补偿，返回新数组（不改输入）。

    cfg（可选）：
      time_scale: float, 默认 1.0。**注意 time 通道量纲**：radar_5frames 的 time 列是
        整数帧索引（[-4..0]，非秒）；10Hz 同步网格下应传 time_scale=0.1（帧索引×0.1s）
        才是真实时间。仅当 time 通道已为秒（t_point-t_cur）时才用默认 1.0。
        误用 1.0 于帧索引会使补偿过量 ~10×、约 7.5% 点被甩出 point_cloud_range。
      use_vr_comp: bool, 默认 False；True 时用 v_r_comp(列_VRC) 代替 v_r(列_VR)
      eps: float, 默认 1e-3（方位角除零保护）
    """
    cfg = cfg or {}
    scale = float(cfg.get('time_scale', 1.0))
    vr_col = _VRC if cfg.get('use_vr_comp', False) else _VR
    eps = float(cfg.get('eps', 1e-3))

    pts = points.copy()
    if pts.shape[0] == 0 or scale == 0.0:
        return pts

    x, y = pts[:, _X], pts[:, _Y]
    vr = pts[:, vr_col]
    t = pts[:, _TIME] * scale

    r = np.sqrt(x * x + y * y) + eps
    vx = vr * (x / r)
    vy = vr * (y / r)
    pts[:, _X] = x - vx * t
    pts[:, _Y] = y - vy * t
    return pts
