"""数据关联 - 欧氏/马氏代价建表 + KM 顶标法最大权匹配."""
from __future__ import annotations
import numpy as np

from .schemas import Cfg, Trk, Obj, Matches


# 不可连边权重: 最大权匹配中不可连边须是最小权 -inf.
_W_NEG_INF: float = -np.inf


class _GapMetric:
    """距离度量 - 单条 (trk, obj) 的关联代价 gap, 越小越优."""

    def __init__(self, gap_type: int, weight: list[float]):
        self.gap_type = gap_type  # 1=欧氏  2=马氏
        self.weight = weight      # [x_weight, y_weight, v_weight]

    def gap(self, trk: Trk, obj: Obj) -> float:
        """计算 trk↔obj 关联代价: 位置项 (欧氏/马氏) + 多普勒加权平方."""
        if self.gap_type == 1:
            # 欧氏: 位置差 + 多普勒差的加权平方和
            pos = (self.weight[0] * (trk.x_m - obj.x) ** 2 +
                   self.weight[1] * (trk.y_m - obj.y) ** 2)
        else:
            # 马氏: x/y 纯马氏距离 (weight 不参与位置项)
            diff = np.array([trk.x_m - obj.x, trk.y_m - obj.y])
            pos = diff @ np.linalg.inv(trk.cov[0:2, 0:2]) @ diff
        dpl = self.weight[2] * (trk.doppler_mps - obj.doppler) ** 2
        return float(pos + dpl)



class _KMSolver:
    """KM 顶标法最大权匹配 - 仅在冲突子图 (match 未定型位) 上增广."""

    def __init__(self, W: np.ndarray, match: np.ndarray):
        self.W = W.astype(np.float64).copy()
        self.n_trk, self.n_obj = W.shape
        self.Li = np.zeros(self.n_trk)            # 行顶标
        self.Lj = np.zeros(self.n_obj)            # 列顶标
        self.match = match.astype(int).copy()     # 预填确定性配对, 冲突位 -1 待增广
        self.Vi = np.zeros(self.n_trk, dtype=bool)
        self.Vj = np.zeros(self.n_obj, dtype=bool)
        self.inc = np.inf                         # 本次 DFS 最小松弛量

    def solve(self) -> np.ndarray:
        """逐行增广."""
        # TODO: 空矩阵短路; Li=max_j W; for i 增广 (失败则松弛顶标重试)
        return self.match

    def _init_labels(self) -> None:
        """Li[i] = max_j W[i,j]."""
        # TODO: self.Li = np.max(self.W, axis=1)
        ...

    def _km_dfs(self, i: int) -> bool:
        """相等子图内找增广路, 回溯翻转匹配."""
        # TODO: 标 Vi; 遍历未访问列记 slack; 相等子图边深搜, 找到写 match
        ...

    def _slack_inc(self) -> float:
        """未访问列上 (Li+Lj-W) 的 min."""
        # TODO: 矢量化取 min
        ...

    def _update_label(self, inc: float) -> None:
        """松弛顶标: Li[Vi]-=inc, Lj[Vj]+=inc."""
        # TODO: self.Li[self.Vi] -= inc; self.Lj[self.Vj] += inc
        ...


class Matcher:
    """关联主类 - 合并 → 建表 → 建图 → KM → 取结果."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.thresh = cfg.MATCH.thresh            # 代价门限
        self.metric = self._build_gap_metric(cfg)

    def _build_gap_metric(self, cfg: Cfg) -> _GapMetric:
        return _GapMetric(gap_type=cfg.MATCH.gap_type, weight=cfg.MATCH.gap_weight)

    def run(self, trks: list[Trk], objs: list[Obj]) -> Matches:
        """统一关联入口."""
        objs = self._merge_objs(trks, objs)          # 合并分裂检测
        W = self._build_cost_table(trks, objs)       # 建代价表
        match = self._build_graph(W)                 # 一对一直接配对, 冲突留 KM
        match = _KMSolver(W, match).solve()          # 冲突子图跑 KM
        return self._post_result(trks, objs, match)  # 汇总输出

    def _merge_objs(self, trks: list[Trk], objs: list[Obj]) -> list[Obj]:
        """用 trk 检查 obj 分裂, 相近则合并."""
        # TODO: 对每个 trk 找马氏距离内的 obj 合并
        ...

    def _build_cost_table(self, trks: list[Trk], objs: list[Obj]) -> np.ndarray:
        """权重矩阵 W: W=thresh-gap, gap>=thresh 填 -inf."""
        # TODO: 双循环填表; 马氏分支预缓存 inv_Σ
        ...

    def _build_graph(self, W: np.ndarray) -> np.ndarray:
        """一对一剪枝: 可连边唯一则直接配对, 冲突 (>1) 留 KM, 0 标未匹配."""
        # TODO: Ti/Tj 计数, 确定性写 match, 冲突位留 -1
        ...

    def _post_result(self, trks: list[Trk], objs: list[Obj], match: np.ndarray) -> Matches:
        """match → matched/unmatched."""
        # TODO: 遍历 match 收集配对与未匹配项
        ...
