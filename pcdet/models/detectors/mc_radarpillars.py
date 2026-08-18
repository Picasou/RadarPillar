from .pointpillar import PointPillar


class MC_RadarPillars(PointPillar):
    # MSR (MC_Single_Radar) 实验线的 RadarPillar 模型载体。
    # 拓扑与 PointPillar 一致: VFE -> BACKBONE_3D(PillarAttention) -> Scatter -> 2D Backbone -> Head,
    # 独立命名仅为区分 MSR 线的 checkpoint/结果归档。配置见 mc/YAML/msr_radarpillar.yaml。
    pass
