from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, get_type_hints

import numpy as np


# ==================================================
# -------------------- 静/动态参数 ------------------
# ==================================================
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

# ==================================================
# -------------------- 帧与点云 --------------------
# ==================================================
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
class Obj:
    """原始目标 - loader 从 bin 加载, 只含能直接读到的字段"""
    id: int = 0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    length: float = 0.0
    width: float = 0.0
    heading: float = 0.0
    type: int = 0
    isghost: int = 0
    ispassable: int = 0


@dataclass
class Objs:
    num: int = 0
    Lst: list = field(default_factory=list)


@dataclass
class Matches:
    """匹配 - match 过程 + 结果。"""
    matched: list = field(default_factory=list)
    unmatched_trks: list = field(default_factory=list)
    unmatched_objs: list = field(default_factory=list)


@dataclass
class TrkState:
    """单帧状态 - 用于 trk.history, 5 个并行数组对应 Trajectory_t.arr_*"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0


@dataclass
class TrkHistory:
    """轨迹历史容器 - 对齐 C Trajectory_t 结构"""
    wt: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dist: float = 0.0
    states: list = field(default_factory=list)


@dataclass
class Trk:
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
    history: TrkHistory         # 4s 隐藏历史 (含 wt/dx/dy/dist + states[Trajectory_t 5 数组])


@dataclass
class Trks:
    num:int
    Lst:list[Trk]


@dataclass
class FRAME:
    gts: GTs
    pts: PTs
    vdd: VDD
    objs: Objs
    voxels: Optional[np.ndarray] = None
    voxel_coords: Optional[np.ndarray] = None
    voxel_num_points: Optional[np.ndarray] = None
    use_lead_xyz: bool = True
    frame_id: str = ''

@dataclass
class FRAMEs:
    num:int
    Lst:list[FRAME]


# ==================================================
# --------------- 配置 (镜像 cfg.yaml) --------------
# ==================================================
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

    @classmethod
    def get_cfg(cls, path: str) -> 'Cfg':
        """从 yaml 文件加载配置."""
        import yaml
        from typing import get_type_hints
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        _MAP = {
            'RUN': CfgRun, 'DATA': CfgData, 'MODEL': CfgModel,
            'FILTER': CfgFilter, 'MATCH': CfgMatch,
            'VISUALIZE': CfgVisualize, 'EVALUATE': CfgEvaluate,
            'MANAGER': CfgManager,
            'vds': CfgVds, 'para': CfgFilterPara,
            'para_kf': CfgFilterParaKf, 'para_abf': dict,
            'para_ekf': dict, 'para_imm': dict,
        }

        def _build(sub_cls, data):
            if sub_cls is dict:
                return data
            hints = get_type_hints(sub_cls)
            kwargs = {}
            for f in fields(sub_cls):
                val = data.get(f.name)
                if val is None:
                    continue
                hint_cls = hints.get(f.name)
                if f.name in _MAP:
                    kwargs[f.name] = _build(_MAP[f.name], val)
                elif hint_cls and hasattr(hint_cls, '__dataclass_fields__'):
                    kwargs[f.name] = _build(hint_cls, val)
                else:
                    kwargs[f.name] = val
            return sub_cls(**kwargs)

        return _build(cls, raw)

    def isvalid(self) -> bool:
        """校验所有配置字段的类型和合法性."""
        self._check_int(self.RUN.mode, 0, 2, 'RUN.mode')
        self._check_int(self.RUN.overlap, 0, 1, 'RUN.overlap')
        self._check_int(self.RUN.delay, 0, None, 'RUN.delay')
        self._check_float_gt(self.RUN.vds.wheelbase_m, 0, 'RUN.vds.wheelbase_m')
        self._check_float(self.RUN.vds.x_pos_m, None, None, 'RUN.vds.x_pos_m')
        self._check_float(self.RUN.vds.y_pos_m, None, None, 'RUN.vds.y_pos_m')
        self._check_float(self.RUN.vds.z_pos_m, None, None, 'RUN.vds.z_pos_m')
        self._check_float_gt(self.RUN.vds.cycle_s, 0, 'RUN.vds.cycle_s')

        # DATA
        if not isinstance(self.DATA.paths, list):
            raise ValueError(f"DATA.paths must be list, got {type(self.DATA.paths).__name__}")
        if len(self.DATA.paths) == 0:
            raise ValueError("DATA.paths cannot be empty")
        for i, p in enumerate(self.DATA.paths):
            if not isinstance(p, str) or not p:
                raise ValueError(f"DATA.paths[{i}] must be non-empty str, got {type(p).__name__}: {p}")

        # MODEL
        if not isinstance(self.MODEL.cfg, str) or not self.MODEL.cfg:
            raise ValueError(f"MODEL.cfg must be non-empty str, got {self.MODEL.cfg}")
        if not isinstance(self.MODEL.ckpt, str) or not self.MODEL.ckpt:
            raise ValueError(f"MODEL.ckpt must be non-empty str, got {self.MODEL.ckpt}")
        self._check_float(self.MODEL.score_thresh, 0, 1, 'MODEL.score_thresh')

        # FILTER
        self._check_int(self.FILTER.type, 1, 4, 'FILTER.type')
        if 'alpha' not in self.FILTER.para.para_abf:
            raise ValueError("FILTER.para.para_abf must contain 'alpha'")
        if 'beta' not in self.FILTER.para.para_abf:
            raise ValueError("FILTER.para.para_abf must contain 'beta'")
        self._check_float_gt(self.FILTER.para.para_abf['alpha'], 0, 'FILTER.para.para_abf.alpha')
        self._check_float_gt(self.FILTER.para.para_abf['beta'], 0, 'FILTER.para.para_abf.beta')
        self._check_int(self.FILTER.para.para_kf.dim, 2, 4, 'FILTER.para.para_kf.dim')
        self._check_matrix(self.FILTER.para.para_kf.q, self.FILTER.para.para_kf.dim, 'FILTER.para.para_kf.q')
        self._check_matrix(self.FILTER.para.para_kf.r, self.FILTER.para.para_kf.dim, 'FILTER.para.para_kf.r')
        if self.FILTER.type >= 3:
            self._check_int(self.FILTER.para.para_ekf.get('dim', 4), 2, 4, 'FILTER.para.para_ekf.dim')
            self._check_matrix(self.FILTER.para.para_ekf.get('q'), self.FILTER.para.para_ekf.get('dim', 4), 'FILTER.para.para_ekf.q')
            self._check_matrix(self.FILTER.para.para_ekf.get('r'), self.FILTER.para.para_ekf.get('dim', 4), 'FILTER.para.para_ekf.r')

        # MATCH
        self._check_int(self.MATCH.gap_type, 1, 2, 'MATCH.gap_type')
        self._check_int(self.MATCH.gap_dim, 2, 4, 'MATCH.gap_dim')
        if not isinstance(self.MATCH.gap_weight, list):
            raise ValueError(f"MATCH.gap_weight must be list, got {type(self.MATCH.gap_weight).__name__}")
        if len(self.MATCH.gap_weight) < self.MATCH.gap_dim:
            raise ValueError(f"MATCH.gap_weight length ({len(self.MATCH.gap_weight)}) < gap_dim ({self.MATCH.gap_dim})")
        for i, w in enumerate(self.MATCH.gap_weight):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"MATCH.gap_weight[{i}] must be >=0 number, got {w}")
        self._check_float(self.MATCH.thresh, 0, None, 'MATCH.thresh')

        # VISUALIZE
        self._check_int(self.VISUALIZE.enable, 0, 1, 'VISUALIZE.enable')
        for k, v in self.VISUALIZE.show.items():
            self._check_int(v, 0, 1, f'VISUALIZE.show.{k}')
        self._check_int(self.VISUALIZE.metrics, 0, 1, 'VISUALIZE.metrics')
        for k, v in self.VISUALIZE.metrics_show.items():
            self._check_int(v, 0, 1, f'VISUALIZE.metrics_show.{k}')

        # EVALUATE
        self._check_int(self.EVALUATE.type, 0, 2, 'EVALUATE.type')
        self._check_int(self.EVALUATE.report, 0, 1, 'EVALUATE.report')
        if not isinstance(self.EVALUATE.template, str) or not self.EVALUATE.template:
            raise ValueError(f"EVALUATE.template must be non-empty str, got {self.EVALUATE.template}")

        # MANAGER
        self._check_int(self.MANAGER.birth_heat, 0, None, 'MANAGER.birth_heat')
        self._check_int(self.MANAGER.death_heat, 0, None, 'MANAGER.death_heat')
        self._check_float_gt(self.MANAGER.dt, 0, 'MANAGER.dt')
        self._check_float_gt(self.MANAGER.history_horizon, 0, 'MANAGER.history_horizon')
        self._check_int(self.MANAGER.adapter.get('smooth', 0), 0, 1, 'MANAGER.adapter.smooth')
        self._check_int(self.MANAGER.adapter.get('markov', 0), 0, 1, 'MANAGER.adapter.markov')

        return True

    def _check_int(self, v, min_val=None, max_val=None, name: str = ''):
        if not isinstance(v, int):
            raise ValueError(f"{name} must be int, got {type(v).__name__}: {v}")
        if min_val is not None and v < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {v}")
        if max_val is not None and v > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {v}")

    def _check_float(self, v, min_val=None, max_val=None, name: str = ''):
        if not isinstance(v, (int, float)):
            raise ValueError(f"{name} must be number, got {type(v).__name__}: {v}")
        if min_val is not None and v < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {v}")
        if max_val is not None and v > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {v}")

    def _check_float_gt(self, v, min_val, name: str = ''):
        self._check_float(v, min_val, None, name)

    def _check_matrix(self, m, dim, name: str = ''):
        if not isinstance(m, list) or len(m) != dim:
            raise ValueError(f"{name} must be {dim}x{dim} matrix, got list of length {len(m) if isinstance(m, list) else 'N/A'}")
        for i, row in enumerate(m):
            if not isinstance(row, list) or len(row) != dim:
                raise ValueError(f"{name}[{i}] must be list of length {dim}, got {len(row) if isinstance(row, list) else 'N/A'}")
            for j, v in enumerate(row):
                if not isinstance(v, (int, float)):
                    raise ValueError(f"{name}[{i}][{j}] must be number, got {type(v).__name__}: {v}")