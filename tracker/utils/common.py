from __future__ import annotations
import os
import numpy as np
import yaml
from easydict import EasyDict

from schemas import VDS, VDD, FRAME, FRAMEs, Trk

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CFG = os.path.normpath(os.path.join(
    THIS_DIR, '..', '..', 'tools', 'cfgs', 'dataset_configs', 'astyx_dataset_radar.yaml'
))
NUM_FEATURES = 5  # x, y, z, rcs, dpl
IS_TURNING_THRESHOLD = 1e-4


def load_data_cfg(path: str = DEFAULT_DATA_CFG) -> EasyDict:
    with open(path, 'r', encoding='utf-8') as f:
        return EasyDict(yaml.safe_load(f))


def frame_to_arrays(frame: FRAME):
    """格式化-> (xy, z, rcs, dpl)"""
    pts = frame.pts.Lst
    if not pts:
        empty2 = np.zeros((0, 2), dtype=np.float32)
        empty1 = np.zeros((0,), dtype=np.float32)
        return empty2, empty1, empty1, empty1
    x   = np.array([p.x_m         for p in pts], dtype=np.float32)
    y   = np.array([p.y_m         for p in pts], dtype=np.float32)
    z   = np.array([p.z_m         for p in pts], dtype=np.float32)
    rcs = np.array([p.rcs         for p in pts], dtype=np.float32)
    dpl = np.array([p.doppler_mps for p in pts], dtype=np.float32)
    return np.stack([x, y], axis=1), z, rcs, dpl


def compensate_frame_forward(xy: np.ndarray, intermediates: list, cycle_s: float):
    """跨帧反向 ego 补偿"""
    for f in intermediates:
        vdd   = f.vdd
        hostv = vdd.speed_ms
        yr    = vdd.yaw_rate
        dx    = hostv * cycle_s
        wt    = yr    * cycle_s
        if abs(yr) < IS_TURNING_THRESHOLD:
            xy[:, 0] -= dx
            continue
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)
        x_old  = xy[:, 0].copy()
        y_old  = xy[:, 1].copy()
        dx_pos = x_old - dx
        dy_pos = y_old
        xy[:, 0] = dx_pos * cos_wt + dy_pos * sin_wt
        xy[:, 1] = dx_pos * (-sin_wt) + dy_pos * cos_wt
    return xy


def accumulate_points(frames: FRAMEs, vds: VDS, accum_frames: int) -> np.ndarray:
    """累积 N 帧点云 → (N, 5)"""
    cur_xy, cur_z, cur_rcs, cur_dpl = frame_to_arrays(frames.Lst[-1])
    if accum_frames > 1:
        idx = len(frames.Lst) - 1
        for k in range(max(0, idx - accum_frames + 1), idx):
            f_k = frames.Lst[k]
            if not f_k.pts.Lst:
                continue
            hist_xy, hist_z, hist_rcs, hist_dpl = frame_to_arrays(f_k)
            hist_xy = compensate_frame_forward(hist_xy, frames.Lst[k + 1 : idx + 1], vds.cycle_s)
            cur_xy  = np.concatenate([cur_xy, hist_xy])
            cur_z   = np.concatenate([cur_z, hist_z])
            cur_rcs = np.concatenate([cur_rcs, hist_rcs])
            cur_dpl = np.concatenate([cur_dpl, hist_dpl])
    if cur_xy.shape[0] == 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)
    return np.stack([cur_xy[:, 0], cur_xy[:, 1], cur_z, cur_rcs, cur_dpl], axis=1)


def crop_range(points: np.ndarray, point_cloud_range) -> np.ndarray:
    """6 维范围裁剪"""
    pcr = point_cloud_range
    keep = (
        (points[:, 0] >= pcr[0]) & (points[:, 0] <= pcr[3]) &
        (points[:, 1] >= pcr[1]) & (points[:, 1] <= pcr[4]) &
        (points[:, 2] >= pcr[2]) & (points[:, 2] <= pcr[5])
    )
    return points[keep]


def compensate_trks(trks: list[Trk], vdd, cycle_s: float):
    """trk 状态 + history 同步补偿 (in-place)"""
    if cycle_s <= 0 or vdd is None or not trks:
        return
    hostv = vdd.speed_ms
    yr    = vdd.yaw_rate
    dx    = hostv * cycle_s
    wt    = yr    * cycle_s
    is_turning = abs(yr) >= IS_TURNING_THRESHOLD
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)
    heading_delta = int(round(np.degrees(wt))) if is_turning else 0

    def _apply(sx, sy, svx, svy):
        if is_turning:
            return (
                (sx - dx) * cos_wt + sy * sin_wt,
                -(sx - dx) * sin_wt + sy * cos_wt,
                svx * cos_wt + svy * sin_wt,
                -svx * sin_wt + svy * cos_wt,
            )
        return sx - dx, sy, svx, svy

    for t in trks:
        xn, yn, vxn, vyn = _apply(t.x_m, t.y_m, t.vx_mps, t.vy_mps)
        t.x_m         = int(round(xn))
        t.y_m         = int(round(yn))
        t.vx_mps      = int(round(vxn))
        t.vy_mps      = int(round(vyn))
        if is_turning:
            t.heading_deg = (t.heading_deg + heading_delta) % 360

        h = t.history
        h.wt   += wt
        h.dx   += dx
        h.dist += abs(dx)

        for s in h.states:
            xn, yn, vxn, vyn = _apply(s.x, s.y, s.vx, s.vy)
            s.x  = xn
            s.y  = yn
            s.vx = vxn
            s.vy = vyn
            if is_turning:
                s.heading = (s.heading + heading_delta) % 360


def compute_cycle_s(vdd_raw_list: list) -> list[float]:
    """从 B_2021.time_100us 帧差计算每帧 cycle_s (秒)。"""
    cycle_s_list = []
    prev_ts = None
    for b in vdd_raw_list:
        ts = b.time_100us
        if prev_ts is None:
            cycle_s_list.append(0.05)
        else:
            dt_ticks = (ts - prev_ts) & 0xFFFF
            cycle_s_list.append(dt_ticks * 1e-4 if dt_ticks != 0 else 0.05)
        prev_ts = ts
    return cycle_s_list


def compensate_state(x, y, vx, vy, vdd, cycle_s):
    """obj 状态从上一周期 ego 系推到当前 ego 系 (4 维)。vdd=None 时透传。"""
    if vdd is None:
        return x, y, vx, vy
    hostv = vdd.speed_ms
    yr    = vdd.yaw_rate
    dx    = hostv * cycle_s
    wt    = yr    * cycle_s

    if abs(yr) < IS_TURNING_THRESHOLD:
        return x - dx, y, vx, vy

    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)
    dx_pos = x - dx
    dy_pos = y
    return (
        dx_pos * cos_wt + dy_pos * sin_wt,
        dx_pos * (-sin_wt) + dy_pos * cos_wt,
        vx * cos_wt + vy * sin_wt,
        vx * (-sin_wt) + vy * cos_wt,
    )
