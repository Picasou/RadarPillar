"""轨迹管理: 航迹集合的创建/删除/输出 (存在概率记分 + 上桌锁存)."""
from __future__ import annotations

import numpy as np

from .schemas import Cfg, Trk, TrkHistory

PROB_INIT = 55    # 存在概率出生分
PROB_GAIN = 10    # 量测到加分/帧
PROB_LOSS = 15    # 断粮扣分/帧


class TrackerManager:
    """
    in : Matches, trks, cycle_s
    out: 原地维护 trks
            create: 未关联观测 → 新航迹 (ID 1-100 最小空闲, 满则跳过)
            delete: 断粮满 death_heat 帧剔除 (唯一死法)
            merge : [待实现]
            output: 存在概率记分 + 三条件满足锁存输出标志 (obstacle_prob)
    """

    def __init__(self, cfg: Cfg) -> None:
        self.birth_heat = cfg.MANAGER.birth_heat   # 上桌存活门槛 (帧)
        self.death_heat = cfg.MANAGER.death_heat   # 断粮删轨阈值 (帧)
        self.prob_output = cfg.MANAGER.prob_output # 上桌存在概率线
        self.birth_pos_std = getattr(cfg.MANAGER, 'birth_pos_std', 0.5)   # 出生位置标准差 (m)
        self.birth_vel_std = getattr(cfg.MANAGER, 'birth_vel_std', 5.0)   # 出生速度标准差 (m/s)

    def run(self, matches, trks: list, cycle_s: float) -> None:
        self._man_create_trks(matches.unmatched_objs, trks, cycle_s)
        self._man_delete_trks(trks)
        self._man_merge_trks(trks)
        self._man_output_trks(trks, cycle_s)

    def _man_create_trks(self, objs: list, trks: list, cycle_s: float) -> None:
        """
        航迹创建: 未关联观测 → 新航迹 (出生计 1 帧), 原地追加 trks
        """
        used = {trk.id for trk in trks}
        for obj in objs:
            trk = self._man_create_utils(obj, used, cycle_s)
            if trk is not None:
                used.add(trk.id)
                trks.append(trk)

    def _man_create_utils(self, obj, used: set, cycle_s: float) -> Trk | None:
        """
        航迹创建工具: 分配 1-100 空闲 ID + 实例化 Trk; ID 满则 None
        """
        tid = next((i for i in range(1, 101) if i not in used), None)
        if tid is None:
            return None
        trk = Trk(
            x_m=obj.x, y_m=obj.y, z_m=0,
            vx_mps=obj.vx, vy_mps=obj.vy,
            doppler_mps=obj.doppler,
            ax_mps2=0, ay_mps2=0,
            heading_deg=obj.heading, yaw_rate_degs=0,
            id=tid, width_m=obj.width, height_m=0, length_m=obj.length, lifetime_s=cycle_s,
            x_std_m=self.birth_pos_std, y_std_m=self.birth_pos_std, z_std_m=0,
            vx_std_mps=self.birth_vel_std, vy_std_mps=self.birth_vel_std,
            ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
            width_std_m=0, height_std_m=0, length_std_m=0,
            heading_std_deg=0, yaw_rate_std_degs=0,
            type=obj.type, type_confi=0, obstacle_prob=0, existence_prob=PROB_INIT,
            motion_status=0, measurement_status=0, passable_status=obj.ispassable,
            rel_vel=0, rel_acc=0,
            cov=np.diag([self.birth_pos_std ** 2, self.birth_pos_std ** 2,
                         self.birth_vel_std ** 2, self.birth_vel_std ** 2]),
            history=TrkHistory(),
        )
        trk.history.push(obj.x, obj.y, 0.0, 0.0, obj.heading)   # 出生状态入史 (帧 0): 速度量测链差分起点
        return trk

    def _man_delete_trks(self, trks: list) -> None:
        """
        航迹删除: 断粮满 death_heat 帧剔除 (唯一死法)
        """
        trks[:] = [t for t in trks if not self._man_delete_utils(t)]

    def _man_delete_utils(self, trk: Trk) -> bool:
        """
        航迹删除工具: 断粮计数达 death_heat 判死
        """
        return trk.measurement_status >= self.death_heat

    def _man_merge_trks(self, trks: list) -> None:
        """
        航迹合并: [待实现]
        """

    def _man_merge_utils(self, trks: list) -> None:
        """
        航迹合并工具: [待实现]
        """

    def _man_output_trks(self, trks: list, cycle_s: float) -> None:
        """
        航迹输出: 记存在概率分 + 三条件满足锁存输出标志 (obstacle_prob)
        """
        for trk in trks:
            self._man_output_utils(trk, cycle_s)

    def _man_output_utils(self, trk: Trk, cycle_s: float) -> None:
        """
        航迹输出工具: 当帧量测 +PROB_GAIN / 断粮 -PROB_LOSS; 没断粮+活满 birth_heat+存在概率达线 → obstacle_prob=1
        """
        trk.existence_prob = min(100, max(0, trk.existence_prob +
                                    (PROB_GAIN if trk.measurement_status == 0 else -PROB_LOSS)))
        if (trk.obstacle_prob == 0
                and trk.measurement_status == 0
                and trk.lifetime_s >= self.birth_heat * cycle_s - 1e-6
                and trk.existence_prob >= self.prob_output):
            trk.obstacle_prob = 1
