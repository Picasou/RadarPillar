from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# 静/动态参数
# ============================================================
@dataclass
class VDS:
    """静态参数 - 轴距、传感器安装位置等。"""
    wheelbase_m: float          # 轴距
    x_pos_m: float              # 传感器安装位置 x
    y_pos_m: float              # 传感器安装位置 y
    z_pos_m: float              # 传感器安装位置 z
    rotation_rad: float         # 传感器水平安装角度
    cycle_s: float              # 采样周期
    oritation: int              # 传感器正反装


@dataclass
class VDD:
    """动态参数 - 车速、档位等。"""
    speed_ms: float             # 车速
    yaw_rate: float             # 横摆角速度
    gear: int                   # 档位

# ============================================================
# 帧与点云
# ============================================================
@dataclass
class PT:
    """单点检测 - 对齐车载雷达 Det_t。"""
    # --- 字节级元信息 ---
    beam: int                   # 波束
    extra_cnt: int              # 附加点计数
    exist_confidence: int       # 存在置信度
    doppler_anti_amb_confi: int # 多普勒解模糊置信度
    # --- 短整型属性 ---
    id: int                     # 点 ID
    flags: int                  # 标志位
    rcs: int                    # 雷达散射截面
    snr: int                    # 信噪比
    frame: int                  # 帧号
    # --- 浮点物理量 ---
    range_m: float              # 距离 (m)
    ang_rad: float              # 方位角 (rad)
    elv_rad: float              # 俯仰角 (rad)
    doppler_mps: float          # 多普勒速度 (m/s)


@dataclass
class PTs:
    num:int
    Lst:list[PT]

@dataclass
class GT:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    length: float
    width: float
    height: float
    heading: float
    type: int
    isghost: int
    ispassable: int

@dataclass
class GTs:
    num:int
    Lst:list[GT]

@dataclass
class FRAME:
    gt: GTs
    pts: PTs
    vdd: VDD

@dataclass
class FRAMEs:
    num:int
    Lst:list[FRAME]
    
    
# ============================================================
# 目标 / 航迹 / 匹配
# ============================================================
@dataclass
class Objs:
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    heading: float
    type: int
    isghost: int
    ispassable: int


@dataclass
class Trks:
    """航迹 - 对齐车载雷达 Trk_t。"""
    # --- 运动学 ---
    x_m: int                    # x 位置 (m)
    y_m: int                    # y 位置 (m)
    z_m: int                    # z 位置 (m)
    vx_mps: int                 # x 速度 (m/s)
    vy_mps: int                 # y 速度 (m/s)
    ax_mps2: int                # x 加速度 (m/s²)
    ay_mps2: int                # y 加速度 (m/s²)
    heading_deg: int            # 航向 (deg)
    yaw_rate_degs: int          # 横摆角速度 (deg/s)

    # --- ID / 尺寸 / 生命 ---
    id: int                     # 航迹 ID
    width_m: int                # 宽 (m)
    height_m: int               # 高 (m)
    length_m: int               # 长 (m)
    lifetime_s: int             # 生命周期 (s)

    # --- 标准差 ---
    x_std_m: int
    y_std_m: int
    z_std_m: int
    vx_std_mps: int
    vy_std_mps: int
    ax_std_mps2: int
    ay_std_mps2: int
    xy_pos_cov: int             # 位置协方差
    xy_vel_cov: int             # 速度协方差
    xy_acc_cov: int             # 加速度协方差
    width_std_m: int
    height_std_m: int
    length_std_m: int
    heading_std_deg: int
    yaw_rate_std_degs: int

    # --- 分类 / 概率 / 状态 ---
    type: int                   # 目标类型
    type_confi: int             # 类型置信度 [0-100]
    obstacle_prob: int          # 障碍概率 [0-100]
    existence_prob: int         # 存在概率 [0-100]
    motion_status: int          # [0:静止 | 1:运动 | 2:慢速]
    measurement_status: int     # [0:coasting | 1:normal]
    passable_status: int        # [0:不可通行 | 1:可通行]
    rel_vel: int                # [0:绝对速度 | 1:相对速度]
    rel_acc: int                # [0:绝对加速度 | 1:相对加速度]
    cov: np.ndarray             # 4x4 协方差
    history: list               # 4s 隐藏历史


@dataclass
class Matches:
    """匹配 - match 过程 + 结果。"""
    matched: list
    unmatched_trks: list[Trks]
    unmatched_objs: list[Objs]


# ============================================================
# 配置 (对齐 cfg.yaml，字段全大写跟 yaml 一级标题)
# ============================================================
@dataclass
class CfgVds:
    """静态参数 - 对齐 RUN.vds。"""
    wheelbase_m: float
    x_pos_m: float
    y_pos_m: float
    z_pos_m: float
    cycle_s: float


@dataclass
class CfgRun:
    """运行配置 - 对齐 RUN。"""
    mode: int                   # 0=display  1=normal  2=regress
    overlap: int                # 0=不覆盖  1=覆盖
    delay: int                  # 雷达滞后实际帧数
    vds: CfgVds


@dataclass
class CfgData:
    """数据配置 - 对齐 DATA。"""
    paths: list[str]


@dataclass
class CfgModel:
    """模型配置 - 对齐 MODEL。"""
    cfg: str                    # 模型结构 yaml
    ckpt: str                   # 权重路径
    score_thresh: float


@dataclass
class CfgFilterParaKf:
    """KF 参数 - 对齐 FILTER.para.para_kf。"""
    dim: int                    # 2=(x/y)  4=(x/y/vx/vy)
    q: float
    r: float


@dataclass
class CfgFilterPara:
    """滤波参数 - 对齐 FILTER.para。"""
    para_abf: dict              # alpha/beta (代码归一化)
    para_kf: CfgFilterParaKf
    para_ekf: dict
    para_imm: dict


@dataclass
class CfgFilter:
    """滤波配置 - 对齐 FILTER。"""
    type: int                   # 1=α-β  2=KF  3=EKF  4=IMM
    para: CfgFilterPara


@dataclass
class CfgMatch:
    """关联配置 - 对齐 MATCH。"""
    gap_type: int               # 1=欧氏  2=马氏
    gap_dim: int                # 2=x/y  3=x/y/z  4=x/y/dpl_gnd
    gap_weight: list[float]
    thresh: float


@dataclass
class CfgVisualize:
    """可视化配置 - 对齐 VISUALIZE。"""
    enable: int
    show: dict                  # points/tracks/objs/gts
    metrics: int
    metrics_show: dict          # 各项指标开关


@dataclass
class CfgEvaluate:
    """性能评估配置 - 对齐 EVALUATE。"""
    type: int                   # 0=off  1=online  2=offline
    report: int
    template: str


@dataclass
class CfgManager:
    """航迹管理配置 - 对齐 MANAGER。"""
    birth_heat: int
    death_heat: int
    dt: float
    history_horizon: float
    adapter: dict               # smooth/markov


@dataclass
class Cfg:
    """配置 - 镜像 cfg.yaml 的 8 大组。"""
    RUN: CfgRun
    DATA: CfgData
    MODEL: CfgModel
    FILTER: CfgFilter
    MATCH: CfgMatch
    VISUALIZE: CfgVisualize
    EVALUATE: CfgEvaluate
    MANAGER: CfgManager
