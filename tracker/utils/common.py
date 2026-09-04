from __future__ import annotations
import os
import numpy as np
import yaml
from easydict import EasyDict

from ..schemas import VDS, VDD, FRAME, Trk

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_CFG = os.path.normpath(os.path.join(
    THIS_DIR, '..', '..', 'tools', 'cfgs', 'dataset', 'astyx_dataset_radar.yaml'
))
NUM_FEATURES = 7  # x, y, z, rcs, v_r, v_r_comp, time
IS_TURNING_THRESHOLD = 1e-4


def load_data_cfg(path: str = DEFAULT_DATA_CFG) -> EasyDict:
    with open(path, 'r', encoding='utf-8') as f:
        return EasyDict(yaml.safe_load(f))


def wrap180(h: float) -> float:
    """
    角度归一: 任意角度 → (-180,180] (heading 落盘 c_int16×100 的量化安全域)
    """
    return -((180.0 - h) % 360.0 - 180.0)


def c_a1_points_convert(frame: FRAME):
    """
    点云格式转换
    (xy, z, rcs, v_r, v_r_comp)
    """
    pts = frame.pts.Lst
    if not pts:
        empty2 = np.zeros((0, 2), dtype=np.float32)
        empty1 = np.zeros((0,), dtype=np.float32)
        return empty2, empty1, empty1, empty1, empty1
    x   = np.array([p.x_m         for p in pts], dtype=np.float32)
    y   = np.array([p.y_m         for p in pts], dtype=np.float32)
    z   = np.array([p.z_m         for p in pts], dtype=np.float32)
    rcs = np.array([p.rcs         for p in pts], dtype=np.float32)
    v_r = np.array([p.doppler_mps for p in pts], dtype=np.float32)

    vdd = frame.vdd
    if vdd is not None:
        ang = np.array([p.ang_rad for p in pts], dtype=np.float32)
        ego_radial = vdd.speed_ms * np.cos(ang)
        v_r_comp = v_r - ego_radial
    else:
        v_r_comp = v_r.copy()

    return np.stack([x, y], axis=1), z, rcs, v_r, v_r_comp


def c_points_compensate(xy: np.ndarray, intermediates: list, cycle_s: float):
    """
    跨帧补偿
    """
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


def c_points_overlay(window: list[FRAME], vds: VDS, accum_frames: int) -> np.ndarray:
    """
    多帧叠加: 在线滚动窗口(最近 accum_frames 帧, 末位为当前帧) → (N, 7) [x,y,z,rcs,v_r,v_r_comp,time]
    """
    idx = len(window) - 1
    cycle_s = vds.cycle_s

    def _feats(f, t):
        xy, z, rcs, v_r, v_r_comp = c_a1_points_convert(f)
        if xy.shape[0] == 0:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32)
        n = xy.shape[0]
        time = np.full(n, t, dtype=np.float32)
        return np.stack([xy[:, 0], xy[:, 1], z, rcs, v_r, v_r_comp, time], axis=1)

    cur = _feats(window[-1], 0.0)
    chunks = [cur] if cur.shape[0] > 0 else []
    if accum_frames > 1:
        for k in range(max(0, idx - accum_frames + 1), idx):
            f_k = window[k]
            if not f_k.pts.Lst:
                continue
            hist = _feats(f_k, -(idx - k) * cycle_s)
            if hist.shape[0] == 0:
                continue
            # 历史帧 xy 反向 ego 补偿到当前帧坐标系
            hist[:, 0:2] = c_points_compensate(hist[:, 0:2], window[k + 1 : idx + 1], cycle_s)
            chunks.append(hist)

    if not chunks:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def c_trk_compensate(trk: Trk, vdd, cycle_s: float):
    """
    单航迹 ego 补偿: 当前 state + 整条 history 同步推到当前 ego 系
    """
    if cycle_s <= 0 or vdd is None:
        return
    wt = vdd.yaw_rate * cycle_s
    heading_delta = int(round(np.degrees(wt))) if abs(vdd.yaw_rate) >= IS_TURNING_THRESHOLD else 0

    # ---- state 段: 当前状态 4 维补偿 + 加速度旋转 + heading ----
    xn, yn, vxn, vyn = c_state_compensate(trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps, vdd, cycle_s)
    trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps = float(xn), float(yn), float(vxn), float(vyn)
    _, _, axn, ayn = c_state_compensate(0.0, 0.0, trk.ax_mps2, trk.ay_mps2, vdd, cycle_s)
    trk.ax_mps2, trk.ay_mps2 = float(axn), float(ayn)
    if heading_delta:
        trk.heading_deg = (trk.heading_deg + heading_delta) % 360

    # ---- history 段: 逐点补偿历史轨迹 + heading ----
    h = trk.history
    h.wt   += wt
    h.dx   += vdd.speed_ms * cycle_s
    h.dist += abs(vdd.speed_ms * cycle_s)
    for j in range(h.tail_idx):
        h.x_history[j], h.y_history[j], h.vx_history[j], h.vy_history[j] = c_state_compensate(
            h.x_history[j], h.y_history[j], h.vx_history[j], h.vy_history[j], vdd, cycle_s)
        if heading_delta:
            h.heading_history[j] = (h.heading_history[j] + heading_delta) % 360


def c_state_compensate(x, y, vx, vy, vdd, cycle_s):
    """
    obj 状态从上一周期 ego 系推到当前 ego 系 (4 维)。vdd=None 时透传。
    """
    if vdd is None:
        return x, y, vx, vy
    dx    = vdd.speed_ms * cycle_s
    wt    = vdd.yaw_rate * cycle_s
    if abs(vdd.yaw_rate) < IS_TURNING_THRESHOLD:
        return x - dx, y, vx, vy
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)
    dx_pos = x - dx
    return (
        dx_pos * cos_wt + y * sin_wt,
        -dx_pos * sin_wt + y * cos_wt,
        vx * cos_wt + vy * sin_wt,
        -vx * sin_wt + vy * cos_wt,
    )
