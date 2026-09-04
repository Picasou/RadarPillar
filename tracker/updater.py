"""航迹更新 - 滤波修正 + 属性维护."""
from __future__ import annotations

import numpy as np
import warnings

from .schemas import Cfg, Matches, Trk
from .filter import Filter

SMOOTH_TAU = 5.0            # 尺寸平滑时间常数 (s), 越大变化越慢
HEADING_ALPHA = 0.2         # 航向向量 α-β: α 向量修正增益
HEADING_BETA = 0.05         # 航向向量 α-β: β 角速度通道增益 (heading 量测脏, 取小)
HEADING_W_CLAMP_DPS = 90.0  # 航向角速度钳位 (deg/s), 防脏量测 windup
ACC_CLAMP_MPS2 = 10.0       # α-β 加速度通道钳位 (m/s²), 防持续偏差下 windup


class VelEstimator:
    """
    速度量测链: history 滑窗头尾差分 → v_meas; meas_dim=2 时 α-β 二次滤波速度 (状态寄存航迹, 模块无自持状态)
    """

    def __init__(self, window_s: float, alpha: float, beta: float) -> None:
        self.window_s = window_s
        self.alpha = alpha
        self.beta = beta

    def measure(self, trk: Trk, cycle_s: float):
        """
        速度量测: history 滑窗头尾差分 → (vx, vy), 不足 2 点返回 None
        """
        h = trk.history
        last = h.tail_idx - 1
        if last < 1:
            return None                          # 仅出生点, 头尾无从差分
        j0 = min(max(last - max(1, int(round(self.window_s / cycle_s))), 0), last - 1)   # 窗头: window_s 内最老点 (保底 2 点)
        h.head_idx = j0
        span = (last - j0) * cycle_s
        return ((h.x_history[last] - h.x_history[j0]) / span,
                (h.y_history[last] - h.y_history[j0]) / span)

    def smooth(self, trk: Trk, v_meas, dt: float) -> None:
        """
        速度二次滤波: α-β 以 v_meas 修正 trk.vx/vy 与加速度通道
        """
        v_meas = np.asarray(v_meas, dtype=float)
        if not trk.vel_init:
            trk.vx_mps, trk.vy_mps = float(v_meas[0]), float(v_meas[1])   # 头尾差分冷启动
            trk.ax_mps2 = trk.ay_mps2 = 0.0
            trk.vel_init = True
            return
        v = np.array([trk.vx_mps, trk.vy_mps]) + np.array([trk.ax_mps2, trk.ay_mps2]) * dt  # predict
        r = v_meas - v
        v = v + self.alpha * r                   # α 修正速度
        a = np.clip(np.array([trk.ax_mps2, trk.ay_mps2]) + (self.beta / dt) * r,
                    -ACC_CLAMP_MPS2, ACC_CLAMP_MPS2)   # β 修正加速度
        trk.vx_mps, trk.vy_mps = float(v[0]), float(v[1])
        trk.ax_mps2, trk.ay_mps2 = float(a[0]), float(a[1])


class Updater:
    """
    航迹更新: matched 滤波修正+属性回填, unmatched 标 coast (原地)
    """

    def __init__(self, cfg: Cfg) -> None:
        self.filter = Filter(cfg)
        tm = cfg.MANAGER.adapter.get('type_markov', {})

        self.smooth = cfg.MANAGER.adapter.get('smooth', 0) == 1
        self.markov = cfg.MANAGER.adapter.get('markov', 0) == 1
        self.class_names = list(tm.get('class_names', ['Car', 'Pedestrian', 'Cyclist']))
        self.n_types = len(self.class_names)
        self.p_stay = float(tm.get('p_stay', 0.95))
        self.accuracy = float(tm.get('accuracy', 0.7))
        P = np.full((self.n_types, self.n_types), (1.0 - self.p_stay) / max(1, self.n_types - 1))
        np.fill_diagonal(P, self.p_stay)               # 自环 p_stay, 其余均分, 行归一
        self.type_P = P / P.sum(axis=1, keepdims=True)
        self.type_states: dict[int, np.ndarray] = {}   # trk.id -> 类型后验
        vel = getattr(cfg, 'VELOCITY', None)           # 旧 cfg/测试桩无此段走默认
        self.vel_enable = (getattr(vel, 'enable', 1) == 1)
        self.vel_est = VelEstimator(window_s=getattr(vel, 'window_s', 1.0),
                                    alpha=getattr(vel, 'alpha', 0.5),
                                    beta=getattr(vel, 'beta', 0.5))

    def run(self, matches: Matches, cycle_s: float) -> None:
        self._udt_maintain(matches, cycle_s)
        self._udt_coasting(matches.unmatched_trks)

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        """
        航迹预测: 剪枝类型后验 + 委托 Filter.predict
        """
        self._prune_type_states(trks)
        self.filter.predict(trks, vdd, cycle_s)

    def reset(self) -> None:
        """
        重置: 清类型后验 + 滤波状态
        """
        self.type_states.clear()
        self.filter.reset()

    def _prune_type_states(self, trks: list[Trk]) -> None:
        """
        类型后验剪枝: 删已消亡 id, 防 ID 复用继承
        """
        alive = {trk.id for trk in trks}
        for tid in [k for k in self.type_states if k not in alive]:
            del self.type_states[tid]

    def _udt_maintain(self, matches: Matches, cycle_s: float) -> None:
        for trk, obj in matches.matched:
            dt_gap = (self._udt_basic(trk, obj, cycle_s) + 1) * cycle_s   # 基础信息回填 → 有效量测间隔
            v_meas = self._udt_velocity(trk, obj, cycle_s)    # 速度量测生成 (须在滤波前: 覆写值供 dim=4 滤波取用)
            self.filter.update(trk, obj, cycle_s)             # 运动状态滤波
            self._udt_vsmooth(trk, v_meas, dt_gap)            # 速度二次滤波
            self._udt_type(trk, obj)                          # 类型更新
            self._udt_size(trk, obj)                          # 尺寸更新
            self._udt_heading(trk, obj, dt_gap)               # 航向更新
            trk.history.push(trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps, trk.heading_deg)   # 滤波后状态入史

    def _udt_basic(self, trk: Trk, obj, cycle_s: float) -> int:
        """
        基础信息回填: doppler/score/lifetime 刷新 + 清 coast 计数, 返回重置前 coast 帧数
        """
        gap = trk.measurement_status
        trk.doppler_mps = getattr(obj, 'doppler', 0.0)
        trk.det_score = float(obj.score)
        trk.measurement_status = 0
        trk.lifetime_s += cycle_s
        return gap

    def _udt_velocity(self, trk: Trk, obj, cycle_s: float):
        """
        速度量测:
        history 滑窗差分 v_meas 覆写 obj.vx/vy
        """
        v = self.vel_est.measure(trk, cycle_s) if self.vel_enable else None
        if v is not None:
            obj.vx, obj.vy = v
        return v

    def _udt_vsmooth(self, trk: Trk, v_meas, dt: float) -> None:
        """
        速度二次滤波:
        meas_dim=2 时启用, 位置归 KF 速度归 α-β
        """
        if self.vel_enable and v_meas is not None and self.filter.meas_dim == 2:
            self.vel_est.smooth(trk, v_meas, dt)

    def _udt_coasting(self, trks: list) -> None:
        for trk in trks:
            trk.measurement_status += 1
            trk.history.push(trk.x_m, trk.y_m, trk.vx_mps, trk.vy_mps, trk.heading_deg)   # 预测状态入史 (逐帧, 索引=帧)

    def _udt_type(self, trk: Trk, obj) -> None:
        """
        类型更新: belief 贝叶斯更新, argmax 输出
        """
        if not self.markov:
            trk.type = obj.type
            trk.type_confi = 100
            return
        z = int(obj.type) - 1                              # 检测器 1-based label → 0-based index
        if not (0 <= z < self.n_types):
            warnings.warn(f"[updater] 未知类别 {obj.type}, 跳过类型更新 (cfg class_names={self.class_names})")
            return
        pi = self.type_states.get(trk.id)
        if pi is None:
            pi = np.full(self.n_types, 1.0 / self.n_types)
        pi = pi @ self.type_P                              # 时间转移
        ll = np.full(self.n_types, (1.0 - self.accuracy) / max(1, self.n_types - 1))
        ll[z] = self.accuracy                              # 量测似然
        pi = pi * ll
        total = float(pi.sum())
        pi = pi / total if total > 0 else np.full(self.n_types, 1.0 / self.n_types)
        self.type_states[trk.id] = pi
        trk.type = int(pi.argmax()) + 1                    # 输出回 1-based (与检测器/真值一致)
        trk.type_confi = int(round(100.0 * float(pi.max())))

    def _udt_size(self, trk: Trk, obj) -> None:
        """
        尺寸更新: 平滑系数随 lifetime 衰减
        """
        if not self.smooth:
            trk.length_m = obj.length
            trk.width_m = obj.width
            return
        alpha = 1.0 / (1.0 + trk.lifetime_s / SMOOTH_TAU)
        trk.length_m += alpha * (obj.length - trk.length_m)
        trk.width_m += alpha * (obj.width - trk.width_m)

    def _udt_heading(self, trk: Trk, obj, dt: float) -> None:
        """
        航向更新: sin/cos 向量 α-β, atan2 回写 deg
        """
        if not self.smooth:
            trk.heading_deg = obj.heading
            trk.yaw_rate_degs = 0.0
            return
        th = np.deg2rad(trk.heading_deg) + np.deg2rad(trk.yaw_rate_degs) * dt   # predict: ω 外推
        ps, pc = np.sin(th), np.cos(th)               # 重投影单位圆, 防外推模长漂
        ms, mc = np.sin(np.deg2rad(obj.heading)), np.cos(np.deg2rad(obj.heading))
        r = np.arctan2(ms * pc - mc * ps,             # 叉/点对融合 = θm-θ̂ 全域角残差 (rad);
                       mc * pc + ms * ps)             # 纯叉积 sin(δ) 在 180° 对置时归零 → α 对消 + β 失感, 卡死不动点
        vs = ps + HEADING_ALPHA * (ms - ps)           # α 向量修正 (模长缩短由 atan2 天然归一)
        vc = pc + HEADING_ALPHA * (mc - pc)
        w = trk.yaw_rate_degs + np.rad2deg(HEADING_BETA / dt) * r   # β: 角残差 rad → 角速度 deg/s
        trk.yaw_rate_degs = float(np.clip(w, -HEADING_W_CLAMP_DPS, HEADING_W_CLAMP_DPS))
        trk.heading_deg = float(np.rad2deg(np.arctan2(vs, vc))) % 360.0
