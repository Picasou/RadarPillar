"""数据关联 - 代价建表 + KM 最大权匹配."""
from __future__ import annotations
import numpy as np

from .schemas import Cfg, Trk, Obj, Matches

_EPS = 1e-6
_UNMATCHED = -1
_MAHA_SCALE = 1.2                    # 马氏椭圆长宽放大系数 (对齐 C maha_covariance 输入)


class Matcher:
    """关联主类 - 建表 → 建图 → KM → 取结果."""

    def __init__(self, cfg: Cfg):
        self.thresh = cfg.MATCH.thresh
        self.gap_type = cfg.MATCH.gap_type            # 1=欧氏  2=马氏
        self.gap_weight = cfg.MATCH.gap_weight        # [x_w, y_w, dpl_w]
        self.gap_dim = cfg.MATCH.gap_dim              # 2=x/y  3=x/y/dpl

    def run(self, trks: list[Trk], objs: list[Obj]) -> Matches:
        W = self._build_cost_table(trks, objs)
        match = self._build_graph(W)
        match = self._km_solve(W, match)
        return self._post_result(trks, objs, match)

    def _maha_inv(self, trk: Trk) -> np.ndarray | None:
        """
        目标外接椭圆协方差逆: 长宽×1.2 + 航向构造 (对齐 C maha_covariance); 退化解返回 None
        """
        cos_h = np.cos(np.deg2rad(trk.heading_deg))
        sin_h = np.sin(np.deg2rad(trk.heading_deg))
        hx = 0.5 * trk.length_m * _MAHA_SCALE
        hy = 0.5 * trk.width_m * _MAHA_SCALE
        r00, r01 = hx * cos_h, -hy * sin_h
        r10, r11 = hx * sin_h, hy * cos_h
        c00, c10 = r00 * r00 + r01 * r01, r00 * r10 + r01 * r11
        d = c00 * (r10 * r10 + r11 * r11) - c10 * c10
        if d <= 0:
            return None
        return np.array([[r10 * r10 + r11 * r11, -c10], [-c10, c00]]) / d

    def _gap(self, trk: Trk, obj: Obj, inv: np.ndarray | None = None) -> float:
        if self.gap_type == 1:
            pos = (self.gap_weight[0] * (trk.x_m - obj.x) ** 2 +
                   self.gap_weight[1] * (trk.y_m - obj.y) ** 2)
        else:
            if inv is None:
                inv = self._maha_inv(trk)
            if inv is None:
                return float('inf')
            diff = np.array([trk.x_m - obj.x, trk.y_m - obj.y])
            pos = diff @ inv @ diff
        if self.gap_dim == 3:
            pos += self.gap_weight[2] * (trk.doppler_mps - obj.doppler) ** 2
        return float(pos)

    def _build_cost_table(self, trks: list[Trk], objs: list[Obj]) -> np.ndarray:
        # W = thresh - gap, gap >= thresh 不可连填 0; 马氏逆每 trk 预计算一次
        W = np.zeros((len(trks), len(objs)))
        invs = [self._maha_inv(t) for t in trks] if self.gap_type == 2 else None
        for i, trk in enumerate(trks):
            for j, obj in enumerate(objs):
                gap = self._gap(trk, obj, invs[i] if invs else None)
                if gap < self.thresh:
                    W[i][j] = self.thresh - gap
        return W

    def _build_graph(self, W: np.ndarray) -> np.ndarray:
        # 可连边唯一 → 直接配对; 冲突 (>1) 留 KM
        n_trk, n_obj = W.shape
        match = np.full(n_obj, _UNMATCHED, dtype=int)
        if n_trk == 0 or n_obj == 0:
            return match
        Ti = np.sum(W > _EPS, axis=1)
        Tj = np.sum(W > _EPS, axis=0)
        for i in range(n_trk):
            for j in range(n_obj):
                if W[i][j] > _EPS and Ti[i] == 1 and Tj[j] == 1:
                    match[j] = i
        return match

    def _km_solve(self, W: np.ndarray, match: np.ndarray) -> np.ndarray:
        n_trk, n_obj = W.shape
        if n_obj == 0:
            return match
        # 冲突子图: Cj = 待解列; Ci = 未被预填占用的行且有待解可连边
        Cj = [j for j in range(n_obj) if match[j] == _UNMATCHED]
        prematched_trks = {match[j] for j in range(n_obj) if match[j] != _UNMATCHED}
        Ci = [i for i in range(n_trk)
              if i not in prematched_trks
              and any(W[i][j] > _EPS and match[j] == _UNMATCHED for j in range(n_obj))]
        if not Ci:
            return match
        Ni, Nj = len(Ci), len(Cj)
        Nji = max(Ni, Nj)
        Ws = np.zeros((Ni, Nji))
        for ii in range(Ni):
            for jj in range(Nj):
                Ws[ii][jj] = W[Ci[ii]][Cj[jj]]
        # 行顶标 = 行最大权, 列顶标 = 0
        Li = np.zeros(Nji)
        Lj = np.zeros(Nji)
        if Nji:
            Li[:Ni] = np.max(Ws, axis=1)
        gm = np.full(Nji, _UNMATCHED, dtype=int)

        def dfs(i, Vi, Vj, slack):
            # 相等子图内找增广路, 回溯翻 gm
            Vi[i] = True
            for j in range(Nji):
                if Vj[j]:
                    continue
                d = Li[i] + Lj[j] - Ws[i][j]
                if d <= _EPS:
                    Vj[j] = True
                    if gm[j] == _UNMATCHED or dfs(gm[j], Vi, Vj, slack):
                        gm[j] = i
                        return True
                elif d < slack[j]:
                    slack[j] = d
            return False

        for i in range(Ni):
            while True:
                Vi = np.zeros(Nji, dtype=bool)
                Vj = np.zeros(Nji, dtype=bool)
                slack = np.full(Nji, np.inf)
                if dfs(i, Vi, Vj, slack):
                    break
                inc = np.min(slack[~Vj])
                Li[Vi] -= inc
                Lj[Vj] += inc

        for jj in range(Nj):
            ii = gm[jj]
            if ii != _UNMATCHED and ii < Ni and Ws[ii][jj] > _EPS:
                match[Cj[jj]] = int(Ci[ii])
        return match

    def _post_result(self, trks: list[Trk], objs: list[Obj], match: np.ndarray) -> Matches:
        res = Matches()
        matched_trk, matched_obj = set(), set()
        for j in range(len(match)):
            i = match[j]
            if 0 <= i < len(trks):
                res.matched.append((trks[i], objs[j]))
                matched_trk.add(i)
                matched_obj.add(j)
        res.unmatched_trks = [trks[i] for i in range(len(trks)) if i not in matched_trk]
        res.unmatched_objs = [objs[j] for j in range(len(objs)) if j not in matched_obj]
        return res
