"""性能评估: MOT 指标集合 (CLEAR/IDF1/HOTA/运动) — 输出航迹 vs GT, per-class 分账."""
from __future__ import annotations

# GT 标注类型 → 检测器类名 (AUTOSIL GTT 预设: 1=轿车 2=行人 4=二轮车 5=卡车 7=静态障碍物)
GT_TYPE_NAMES = {1: 'Car', 2: 'Pedestrian', 4: 'Cyclist', 5: 'Truck', 7: 'StaticObject'}

# 指标集合: 报告输出项, 按 METRICS.show 开关过滤 (AMOTA 族无 score 来源, 不做)
METRIC_KEYS = ('tp', 'fp', 'fn', 'ids', 'frag', 'mota', 'motp',
               'idf1', 'deta', 'assa', 'hota', 'loca',
               'vae', 'vne', 'vaie', 'vir', 'vse', 'vde')


class Evaluator:
    """
    in : online→(FRAME, 本帧输出航迹) / on_seq_end→序列名 / evaluate→history[seq[(gts, trks)]]
    out: METRIC_KEYS 指标集合 (total + per-class) 控制台 + 报告文件
    """

    def __init__(self, cfg, class_names: list[str] | None = None) -> None:
        """
        初始化: 评估配置 (匹配门限/指标开关/报告开关) + 类名对齐 + 状态清零
        """
        ...

    # ---- online (eval_mode=1) ----

    def online(self, frame, trks: list) -> None:
        """
        在线累计: 逐帧记账, 结果由 on_seq_end 打印
        """
        ...

    def on_seq_end(self, seq_name: str) -> None:
        """
        序列收尾: 打印本序列累计并清零
        """
        ...

    # ---- offline (eval_mode=2) ----

    def evaluate(self, history: list) -> dict:
        """
        离线评估: history[seq[(gts, trks)]] -> per-seq/per-class + 总计, 打印 + 报告落盘
        """
        ...

    # ---- 指标集合 ----

    def match_frame(self, gts: list, trks: list):
        """
        单帧匹配: 贪心最近邻 + 类别约束 + 距离门限 (单测测试缝, 保留)
        返回 (matched[(gt,trk,dist)], unmatched_trks, unmatched_gts)
        """
        ...

    def _step(self, gts: list, trks: list) -> None:
        """
        单帧记账: 匹配配对 -> 各指标原料累计
        """
        ...

    def _finalize(self) -> dict:
        """
        汇总: 累计原料 -> METRIC_KEYS 指标集合
        """
        ...
