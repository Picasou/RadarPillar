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
    # --- 直角坐标 (loader 极坐标→直角后填充) ---
    x_m: float = 0.0            # x 位置 (m)
    y_m: float = 0.0            # y 位置 (m)
    z_m: float = 0.0            # z 位置 (m)


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
    id: int = 0                  # 标注持久编号 (gt bin 加载填, IDSW/Frag 依赖)
    type_confi: int = 0          # 分类置信度 [0-100]
    is_attention: int = 1        # 可关注目标标记 [0|1]
    point_count: int = 0         # 落 BEV 框内雷达点数 [0-255]
    point_quality: int = 0       # 点密度 4*count/(w*l), 钳 [0-100]

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
    doppler: float = 0.0          # 多普勒(径向速度) m/s (loader/detector 填)
    length: float = 0.0
    width: float = 0.0
    heading: float = 0.0
    type: int = 0
    isghost: int = 0
    ispassable: int = 0
    score: float = 0.0            # 检测置信度 (detector 填, loader 路径默认 0)
    z: float = 0.0                # 高度中心 (m) (detector 填 box[2], loader 回灌填)
    height: float = 0.0           # 高度 (m) (detector 填 box[5], loader 回灌填)


@dataclass
class Objs:
    num: int = 0
    Lst: list = field(default_factory=list)


@dataclass
class Matches:
    """
    匹配 - match 过程 + 结果。

    契约:
        matched        : list[tuple[Trk, Obj]] - 关联成功的 (航迹, 观测) 对
        unmatched_trks : list[Trk]             - 未关联航迹
        unmatched_objs : list[Obj]             - 未关联观测
    """
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


HISTORY_LEN = 64    # 逐帧入史 @10Hz = 6.4s, 覆盖 window_s 滑窗 + history_horizon 余量
@dataclass
class TrkHistory:
    """轨迹历史容器 - 对齐 C Trajectory_t 结构; 逐帧写滤波后状态, 索引=帧 (Δt=帧差×cycle_s)"""
    wt: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dist: float = 0.0
    head_idx: int = 0    # 差分窗头指针 (速度量测链按 window_s 维护, [head_idx, tail_idx) 为窗)
    tail_idx: int = 0    # 写入游标 (线性滑窗语义 [0, tail_idx))
    x_history:      np.ndarray = field(default_factory=lambda: np.zeros(HISTORY_LEN))
    y_history:      np.ndarray = field(default_factory=lambda: np.zeros(HISTORY_LEN))
    vx_history:     np.ndarray = field(default_factory=lambda: np.zeros(HISTORY_LEN))
    vy_history:     np.ndarray = field(default_factory=lambda: np.zeros(HISTORY_LEN))
    heading_history: np.ndarray = field(default_factory=lambda: np.zeros(HISTORY_LEN))

    def push(self, x: float, y: float, vx: float, vy: float, heading: float) -> None:
        """
        历史追加: 滤波后状态 (x,y,vx,vy,heading) 逐帧写入队尾, 满则整体左移一格 (head_idx 随移)
        """
        if self.tail_idx >= HISTORY_LEN:
            for arr in (self.x_history, self.y_history, self.vx_history,
                        self.vy_history, self.heading_history):
                arr[:-1] = arr[1:]
            self.tail_idx = HISTORY_LEN - 1
            self.head_idx = max(0, self.head_idx - 1)
        i = self.tail_idx
        self.x_history[i] = x
        self.y_history[i] = y
        self.vx_history[i] = vx
        self.vy_history[i] = vy
        self.heading_history[i] = heading
        self.tail_idx = i + 1
                                 


@dataclass
class Trk:
    """航迹 - 对齐车载雷达 Trk_t。"""
    # --- 运动学 (float: 物理量需全精度) ---
    x_m: float                  # x 位置 (m)
    y_m: float                  # y 位置 (m)
    z_m: float                  # z 位置 (m)
    vx_mps: float               # x 速度 (m/s)
    vy_mps: float               # y 速度 (m/s)
    doppler_mps: float          # 多普勒(径向速度) m/s
    ax_mps2: float              # x 加速度 (m/s²)
    ay_mps2: float              # y 加速度 (m/s²)
    heading_deg: float          # 航向 (deg)
    yaw_rate_degs: float        # 横摆角速度 (deg/s)

    # --- ID / 尺寸 / 生命 ---
    id: int                     # 航迹 ID (枚举整数)
    width_m: float              # 宽 (m)
    height_m: float             # 高 (m)
    length_m: float             # 长 (m)
    lifetime_s: float           # 生命周期 (s)

    # --- 标准差 (float) ---
    x_std_m: float
    y_std_m: float
    z_std_m: float
    vx_std_mps: float
    vy_std_mps: float
    ax_std_mps2: float
    ay_std_mps2: float
    xy_pos_cov: float           # 位置协方差
    xy_vel_cov: float           # 速度协方差
    xy_acc_cov: float           # 加速度协方差
    width_std_m: float
    height_std_m: float
    length_std_m: float
    heading_std_deg: float
    yaw_rate_std_degs: float

    # --- 分类 / 概率 / 状态 ---
    type: int                 # 目标类型 (枚举整数)
    type_confi: int           # 类型置信度 [0-100]
    obstacle_prob: int        # 输出锁存 [0:未上桌 | 1:输出中]
    existence_prob: int       # 存在概率 [0-100]
    motion_status: int          # [0:静止 | 1:运动 | 2:慢速] (枚举)
    measurement_status: int     # 连续未量测帧数 (0=当帧有量测)
    passable_status: int        # [0:不可通行 | 1:可通行] (枚举)
    rel_vel: int                # [0:绝对速度 | 1:相对速度] (标志)
    rel_acc: int                # [0:绝对加速度 | 1:相对加速度] (标志)
    cov: np.ndarray             # 4x4 协方差
    history: TrkHistory         # 4s 隐藏历史 (含 wt/dx/dy/dist + states[Trajectory_t 6 数组])
    vel_init: bool = False      # 速度量测链冷启动标志 (Python 侧工作字段, 不入 C 契约)
    det_score: float = 0.0      # 最近检测置信度 (Python 侧工作字段, 不入 C 契约; AMOTA 扫描用)


@dataclass
class Trks:
    num:int
    Lst:list[Trk]


@dataclass
class FrameProc:
    """处理层产物 - 各模块按需写入的中间结果, 与原始数据物理隔离。"""
    points: Optional[np.ndarray] = None   # (N, 7) [x,y,z,rcs,v_r,v_r_comp,time] 累积+裁剪后 (preprocessor 填)
    voxels: Optional[np.ndarray] = None   # (V, P, C) detector 体素化后填
    voxel_coords: Optional[np.ndarray] = None
    voxel_num_points: Optional[np.ndarray] = None
    use_lead_xyz: bool = True


@dataclass
class FRAME:
    """数据层 - loader 产出的原始帧, 只含输入契约。"""
    gts: GTs
    pts: PTs
    vdd: VDD
    objs: Objs
    proc: FrameProc = field(default_factory=FrameProc)


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
    save: int                   # 0=不保存  1=保存(航迹结果落盘)
    overlap: int                # 航迹bin落盘(写回数据源 radar.default/0200|0201): 0=另存(mode=1→00001, mode=2→00002, 同名覆盖)  1=覆盖原始 00000(可视化图恒覆盖)
    delay: int                  # 雷达滞后实际帧数 (点云 i 配 vdd[i-delay], 对齐 ego 环形缓冲 use_idx)
    vds: CfgVds
    accum_frames: int = 1       # 点云叠加帧数 (1=不叠加)


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
    device: str = 'auto'        # auto | cuda | cpu (auto=有 GPU 走 cuda 否则 cpu)


@dataclass
class CfgFilterParaKf:
    """KF 参数 - 对齐 FILTER.para.para_kf。dim 是量测维(状态恒 4 维)。"""
    dim: int                    # 量测维: 2=仅(x/y)  4=(x/y/vx/vy); 状态恒 [x,y,vx,vy]
    q_acc: float = 5.0          # 过程噪声加速度标准差 (m/s²), CV 离散白噪声模型按 dt 展开为 Q
    r: float = 0.5              # 量测噪声协方差 (标量→对角阵, dim×dim)


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
    gap_dim: int                # 2=x/y  3=x/y/dpl
    gap_weight: list[float]
    thresh: float


@dataclass
class CfgVisual:
    """可视化配置 - 对齐 VISUAL。"""
    enable: int                 # 总开关: 0=完全不出图  1=可视化
    save: list                  # 落盘格式数组: 1=GIF  2=PNG序列  3=MP4; 空数组=不落盘
    show: dict                  # points/tracks/objs/gts
    range: Optional[list] = None    # 固定坐标 [x_lo, x_hi, y_lo, y_hi]
    test_val: Optional[list] = None    # test/val 序列名列表(命中加黑边框)
    cam_rotate: int = 0             # 相机画面转正: 0=不转 1=逆时针90° 2=180° 3=顺时针90°


@dataclass
class CfgMetrics:
    """指标显示配置 - 对齐 METRICS。"""
    enable: int                 # 指标面板总开关: 0=不显示  1=显示
    show: dict                  # 各项指标开关


@dataclass
class CfgEvaluate:
    """性能评估配置 - 对齐 EVALUATE。"""
    type: int                   # 0=off  1=online(逐帧记账)  2=offline(完整报告)
    report: int
    template: str
    match_dist: float = 2.0     # 评估配对距离门限 (m)


@dataclass
class CfgManager:
    """航迹管理配置 - 对齐 MANAGER。"""
    birth_heat: int
    death_heat: int
    dt: float
    history_horizon: float
    adapter: dict               # smooth/markov
    prob_output: int = 80       # 上桌存在概率线
    birth_pos_std: float = 0.5  # 出生位置标准差 (m), P₀ 对角用
    birth_vel_std: float = 5.0  # 出生速度标准差 (m/s), P₀ 对角用
    merge_dist: float = 1.0     # 重复航迹合并距离门限 (m), 中心距小于此值合并 (≤0 关闭)
    birth_min_range: float = 2.0  # 近距建轨抑制 (m), 距 ego 小于此值的观测不建轨 (≤0 关闭)


@dataclass
class CfgVelocity:
    """速度量测链配置 - 对齐 VELOCITY (历史位置滑窗头尾差分 + α-β 平滑 → KF 速度量测)。"""
    enable: int = 1             # 1=启用速度量测链 (0=沿用检测 vx/vy 占位 0)
    window_s: float = 1.0       # 头尾差分滑窗时长 (s)
    alpha: float = 0.5          # α-β 速度平滑增益 α
    beta: float = 0.5           # α-β 速度平滑增益 β (加速度通道)


@dataclass
class Cfg:
    """配置 - 镜像 cfg.yaml 的 10 大组。"""
    RUN: CfgRun
    DATA: CfgData
    MODEL: CfgModel
    FILTER: CfgFilter
    MATCH: CfgMatch
    VISUAL: CfgVisual
    METRICS: CfgMetrics
    EVALUATE: CfgEvaluate
    MANAGER: CfgManager
    VELOCITY: CfgVelocity = field(default_factory=CfgVelocity)

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
            'VISUAL': CfgVisual, 'METRICS': CfgMetrics,
            'EVALUATE': CfgEvaluate,
            'MANAGER': CfgManager, 'VELOCITY': CfgVelocity,
            'vds': CfgVds, 'para': CfgFilterPara,
            'para_kf': CfgFilterParaKf, 'para_abf': dict,
            'para_ekf': dict, 'para_imm': dict,
        }

        def _build(sub_cls, data):
            if sub_cls is dict:
                return data
            try:
                hints = get_type_hints(sub_cls)
            except Exception:
                hints = {}
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
        self._check_int(self.RUN.save, 0, 1, 'RUN.save')
        self._check_int(self.RUN.overlap, 0, 1, 'RUN.overlap')
        self._check_int(self.RUN.delay, 0, None, 'RUN.delay')
        self._check_int_gt(self.RUN.accum_frames, 0, 'RUN.accum_frames')
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
        if self.MODEL.device not in ('auto', 'cuda', 'cpu'):
            raise ValueError(f"MODEL.device must be 'auto'|'cuda'|'cpu', got {self.MODEL.device}")

        # FILTER
        self._check_int(self.FILTER.type, 1, 4, 'FILTER.type')
        if 'alpha' not in self.FILTER.para.para_abf:
            raise ValueError("FILTER.para.para_abf must contain 'alpha'")
        if 'beta' not in self.FILTER.para.para_abf:
            raise ValueError("FILTER.para.para_abf must contain 'beta'")
        self._check_float_gt(self.FILTER.para.para_abf['alpha'], 0, 'FILTER.para.para_abf.alpha')
        self._check_float_gt(self.FILTER.para.para_abf['beta'], 0, 'FILTER.para.para_abf.beta')
        self._check_int(self.FILTER.para.para_kf.dim, 2, 4, 'FILTER.para.para_kf.dim')
        # 状态恒 4 维 → Q 由 q_acc 按 dt 展开; R 按量测维 dim
        self._check_float_gt(self.FILTER.para.para_kf.q_acc, 0, 'FILTER.para.para_kf.q_acc')
        self._check_matrix(self.FILTER.para.para_kf.r, self.FILTER.para.para_kf.dim, 'FILTER.para.para_kf.r')
        if self.FILTER.type >= 3:
            self._check_int(self.FILTER.para.para_ekf.get('dim', 4), 2, 4, 'FILTER.para.para_ekf.dim')
            self._check_float_gt(self.FILTER.para.para_ekf.get('q_acc', 5.0), 0, 'FILTER.para.para_ekf.q_acc')
            self._check_matrix(self.FILTER.para.para_ekf.get('r'), self.FILTER.para.para_ekf.get('dim', 4), 'FILTER.para.para_ekf.r')

        # MATCH
        self._check_int(self.MATCH.gap_type, 1, 2, 'MATCH.gap_type')
        self._check_int(self.MATCH.gap_dim, 2, 3, 'MATCH.gap_dim')
        if not isinstance(self.MATCH.gap_weight, list):
            raise ValueError(f"MATCH.gap_weight must be list, got {type(self.MATCH.gap_weight).__name__}")
        # gap() 索引 weight[0/1] (位置x/y), weight[2] (多普勒) 仅 gap_dim=3 用; 须 >=3
        if len(self.MATCH.gap_weight) < 3:
            raise ValueError(f"MATCH.gap_weight length ({len(self.MATCH.gap_weight)}) must be >= 3 (x/y/doppler)")
        for i, w in enumerate(self.MATCH.gap_weight):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"MATCH.gap_weight[{i}] must be >=0 number, got {w}")
        self._check_float(self.MATCH.thresh, 0, None, 'MATCH.thresh')

        # VISUAL
        self._check_int(self.VISUAL.enable, 0, 1, 'VISUAL.enable')
        s = self.VISUAL.save
        if not isinstance(s, list) or not all(x in (1, 2, 3) for x in s):
            raise ValueError(f"VISUAL.save 须为 [1,2,3] 子集列表(1=GIF 2=PNG 3=MP4,空=不落盘), got {s}")
        if self.VISUAL.range is not None:
            r = self.VISUAL.range
            if not isinstance(r, list) or len(r) != 4 or \
                    not all(isinstance(x, (int, float)) for x in r):
                raise ValueError(f"VISUAL.range 须为 [x_lo, x_hi, y_lo, y_hi] 数值列表, got {r}")
            if not (r[0] < r[1] and r[2] < r[3]):
                raise ValueError(f"VISUAL.range 须 lo < hi, got {r}")
        if self.VISUAL.test_val is not None:
            tv = self.VISUAL.test_val
            if not isinstance(tv, list) or not all(isinstance(s, str) for s in tv):
                raise ValueError(f"VISUAL.test_val 须为序列名字符串列表, got {tv}")
        self._check_int(self.VISUAL.cam_rotate, 0, 3, 'VISUAL.cam_rotate')
        for k, v in self.VISUAL.show.items():
            self._check_int(v, 0, 1, f'VISUAL.show.{k}')
        # METRICS
        self._check_int(self.METRICS.enable, 0, 1, 'METRICS.enable')
        for k, v in self.METRICS.show.items():
            self._check_int(v, 0, 1, f'METRICS.show.{k}')

        # EVALUATE
        self._check_int(self.EVALUATE.type, 0, 2, 'EVALUATE.type')
        self._check_float_gt(self.EVALUATE.match_dist, 0, 'EVALUATE.match_dist')
        self._check_int(self.EVALUATE.report, 0, 1, 'EVALUATE.report')
        if not isinstance(self.EVALUATE.template, str) or not self.EVALUATE.template:
            raise ValueError(f"EVALUATE.template must be non-empty str, got {self.EVALUATE.template}")

        # MANAGER
        self._check_int(self.MANAGER.birth_heat, 0, None, 'MANAGER.birth_heat')
        self._check_int(self.MANAGER.death_heat, 1, None, 'MANAGER.death_heat')   # ≥1: 0 会连当帧刚量测航迹一并删除
        self._check_float(self.MANAGER.merge_dist, 0, None, 'MANAGER.merge_dist')
        self._check_float(self.MANAGER.birth_min_range, 0, None, 'MANAGER.birth_min_range')
        self._check_int(self.MANAGER.prob_output, 0, 100, 'MANAGER.prob_output')
        self._check_float_gt(self.MANAGER.dt, 0, 'MANAGER.dt')
        self._check_float_gt(self.MANAGER.history_horizon, 0, 'MANAGER.history_horizon')
        self._check_int(self.MANAGER.adapter.get('smooth', 0), 0, 1, 'MANAGER.adapter.smooth')
        self._check_int(self.MANAGER.adapter.get('markov', 0), 0, 1, 'MANAGER.adapter.markov')
        tm = self.MANAGER.adapter.get('type_markov', {})
        cn = tm.get('class_names', ['Car', 'Pedestrian', 'Cyclist'])
        if not isinstance(cn, list) or not cn or not all(isinstance(x, str) and x for x in cn):
            raise ValueError(f"MANAGER.adapter.type_markov.class_names 须为非空字符串列表, got {cn}")
        self._check_float(tm.get('p_stay', 0.95), 0, 1, 'MANAGER.adapter.type_markov.p_stay')
        self._check_float(tm.get('accuracy', 0.7), 0, 1, 'MANAGER.adapter.type_markov.accuracy')
        self._check_float_gt(self.MANAGER.birth_pos_std, 0, 'MANAGER.birth_pos_std')
        self._check_float_gt(self.MANAGER.birth_vel_std, 0, 'MANAGER.birth_vel_std')

        # VELOCITY
        self._check_int(self.VELOCITY.enable, 0, 1, 'VELOCITY.enable')
        self._check_float_gt(self.VELOCITY.window_s, 0, 'VELOCITY.window_s')
        self._check_float_gt(self.VELOCITY.alpha, 0, 'VELOCITY.alpha')
        self._check_float_gt(self.VELOCITY.beta, 0, 'VELOCITY.beta')

        return True

    def _check_int(self, v, min_val=None, max_val=None, name: str = ''):
        if not isinstance(v, int):
            raise ValueError(f"{name} must be int, got {type(v).__name__}: {v}")
        if min_val is not None and v < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {v}")
        if max_val is not None and v > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {v}")

    def _check_int_gt(self, v, min_val, name: str = ''):
        self._check_int(v, min_val, None, name)

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