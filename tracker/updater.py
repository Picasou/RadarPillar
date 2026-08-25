"""航迹更新 - 滤波修正 + 属性维护 (measurement_status / 观测属性回填)."""
from __future__ import annotations

import numpy as np
import warnings

from .schemas import Cfg, Matches, Trk
from .filter import Filter

SMOOTH_TAU = 5.0            # 尺寸平滑时间常数 (s), 越大变化越慢
HEADING_SMOOTH_ALPHA = 0.2  # 航向平滑系数


class Updater:
    """
    in : Matches, cycle_s
    out: Update trks (原地)
            matched 滤波修正+观测回填;
            unmatched_trks 标 coast
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

    def run(self, matches: Matches, cycle_s: float) -> None:
        self._udt_miantain(matches, cycle_s)
        self._udt_coasting(matches.unmatched_trks)

    def predict(self, trks: list[Trk], vdd, cycle_s: float) -> None:
        """
        航迹预测: 先按存活剪枝类型后验, 再委托 Filter.predict (ego 补偿 + 状态外推)
        """
        self._prune_type_states(trks)
        self.filter.predict(trks, vdd, cycle_s)

    def reset(self) -> None:
        """
        更新器重置: 清类型后验记忆 + 委托滤波清状态 (序列边界调用)
        """
        self.type_states.clear()
        self.filter.reset()

    def _prune_type_states(self, trks: list[Trk]) -> None:
        """
        类型后验剪枝: 删除不在存活列表的 id (防 ID 复用继承旧航迹类型)
        """
        alive = {trk.id for trk in trks}
        for tid in [k for k in self.type_states if k not in alive]:
            del self.type_states[tid]

    def _udt_miantain(self, matches: Matches, cycle_s: float) -> None:
        for trk, obj in matches.matched:
            trk.doppler_mps = getattr(obj, 'doppler', 0.0)   # dpl update
            trk.measurement_status = 0
            trk.lifetime_s += cycle_s          # life_cnt++
            self.filter.update(trk, obj, cycle_s)    # state update
            self._udt_type(trk, obj)           # type update
            self._udt_size(trk, obj)           # size update
            self._udt_heading(trk, obj)        # heading update

    def _udt_coasting(self, trks: list) -> None:
        for trk in trks:
            trk.measurement_status += 1

    def _udt_type(self, trk: Trk, obj) -> None:
        """类型更新: belief 贝叶斯更新 (转移×量测似然), argmax 输出"""
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
        """尺寸更新: 平滑系数随存在时间衰减, 越久变化越慢"""
        if not self.smooth:
            trk.length_m = obj.length
            trk.width_m = obj.width
            return
        alpha = 1.0 / (1.0 + trk.lifetime_s / SMOOTH_TAU)
        trk.length_m += alpha * (obj.length - trk.length_m)
        trk.width_m += alpha * (obj.width - trk.width_m)

    def _udt_heading(self, trk: Trk, obj) -> None:
        """航向更新: 最短弧平滑去周期 (deg), 抑制盒朝向抖动"""
        if not self.smooth:
            trk.heading_deg = obj.heading
            return
        diff = (obj.heading - trk.heading_deg + 180.0) % 360.0 - 180.0
        trk.heading_deg = (trk.heading_deg + HEADING_SMOOTH_ALPHA * diff) % 360.0
