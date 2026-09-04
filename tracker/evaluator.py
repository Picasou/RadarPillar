"""性能评估: MOT 指标集合 (CLEAR/IDF1/HOTA/运动) — 输出航迹 vs GT, per-class 分账."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# GT 标注类型 → 检测器类名
GT_TYPE_NAMES = {1: 'Car', 2: 'Pedestrian', 4: 'Cyclist', 5: 'Truck', 7: 'StaticObject'}

# 指标集合: 报告输出项, 按 METRICS.show 开关过滤
METRIC_KEYS = ('tp', 'fp', 'fn', 'ids', 'frag', 'mota', 'motp', 'mt', 'ml',
               'idf1', 'deta', 'assa', 'hota', 'loca', 'amota', 'amotp',
               'vae', 'vne', 'vaie', 'vir', 'vse', 'vde')

_VEL_GATE = 0.5        # 速度方向指标门限: |v_gt| 低于此值不计 (静止目标角度无意义)
_SG_WIN = 7            # VSE Savitzky-Golay 窗口 (奇数)
_VDE_MAXLAG = 5        # VDE 最大搜索时移 (帧)
_CC_TIE = 9            # VDE 相关系数并列判定小数位 (吸收浮点噪声)


def _new_acc() -> dict:
    """
    空账本: 计数/求和原料 (总量与 per-class 同构), 含全局指标折算量 (序列末累入)
    """
    return {'tp': 0, 'fp': 0, 'fn': 0, 'ids': 0, 'frag': 0, 'gt_total': 0,
            'dist_sum': 0.0, 'vel_cnt': 0, 'vae_sum': 0.0, 'vne_cnt': 0, 'vne_sum': 0.0,
            'vaie_sum': 0.0, 'vir_cnt': 0, 'vse_sum': 0.0, 'vse_cnt': 0,
            'vde_sum': 0.0, 'vde_cnt': 0, 'idtp': 0, 'assa_w': 0.0,
            'gt_n': 0, 'mt': 0, 'ml': 0}


def _merge_acc(dst: dict, src: dict) -> None:
    """
    账本合并: 同构账逐键累加 (序列 → 数据集, 纯标量 O(1))
    """
    for k, v in src.items():
        dst[k] += v


def _runs(series: list) -> list:
    """
    连续段切分: 帧号断续 (漏配帧) 处切段, 供 VSE/VDE 时序计算
    """
    runs, cur = [], []
    for row in series:
        if cur and row[0] != cur[-1][0] + 1:
            runs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        runs.append(cur)
    return runs


def snap_trk(t):
    """
    航迹评估快照: 仅评估所需 7 字段 (替代 deepcopy, 内存约 1/10)
    """
    return SimpleNamespace(id=t.id, type=t.type, x_m=t.x_m, y_m=t.y_m,
                           vx_mps=t.vx_mps, vy_mps=t.vy_mps,
                           score=float(getattr(t, 'det_score', 0.0)))


class Evaluator:
    """
    in : online→(FRAME, 本帧输出航迹) / on_seq_start·on_seq_end→序列路径 / evaluate→history[(seq_path, [(gts, trks)])]
    out: METRIC_KEYS 指标集合 (per-seq + dataset, total + per-class) 控制台 + 报告文件 + 曲线图
    注: dataset 级为全帧合并 (micro-pooling), 非 per-seq 平均; 全局指标 (IDF1/HOTA) 逐帧只记账, 序列末折算
    """

    # 指标段落: 块打印时按段聚合, 段内各项受 METRICS.show 过滤
    _SECTIONS = (('tp', 'fp', 'fn', 'ids', 'frag'),
                 ('mota', 'motp'),
                 ('mt', 'ml'),
                 ('idf1',),
                 ('deta', 'assa', 'hota', 'loca'),
                 ('amota', 'amotp'),
                 ('vae', 'vne', 'vaie', 'vir', 'vse', 'vde'))

    def __init__(self, cfg, class_names: list[str] | None = None) -> None:
        """
        初始化: 评估配置 (匹配门限/指标开关/报告开关) + 类名对齐 + 状态清零
        """
        self.cfg = cfg
        if class_names is None:                 # 无检测器 (mode=0): 类名取自模型配置
            import yaml
            with open(cfg.MODEL.cfg, 'r', encoding='utf-8') as f:
                class_names = (yaml.safe_load(f) or {}).get('CLASS_NAMES')
        self.class_names = list(class_names or [])
        self.match_dist = float(getattr(cfg.EVALUATE, 'match_dist', 2.0))
        self.do_report = int(getattr(cfg.EVALUATE, 'report', 0)) == 1
        self.show = dict(getattr(cfg.METRICS, 'show', {}) or {})
        self.template = str(getattr(cfg.EVALUATE, 'template', 'default'))
        self.dt = float(getattr(cfg.MANAGER, 'dt', 0.1))
        self.show_panel = int(getattr(cfg.METRICS, 'enable', 1)) == 1

        # GT 类型 → 检测 label: 数字类名按数值, 其余按名字; 无对应类不入表
        self.gt2label: dict[int, int] = {}
        for i, name in enumerate(self.class_names):
            if name.strip().isdigit():
                self.gt2label[int(name)] = i + 1
            else:
                for gt_t, n in GT_TYPE_NAMES.items():
                    if n == name:
                        self.gt2label[gt_t] = i + 1

        self._reset_dataset()
        self._reset_seq()

    # ---- 状态重置 ----

    def _reset_dataset(self) -> None:
        """
        数据集级重置: 标量账 (无原始配对表/速度序列驻留, 内存 O(1)) + 报告缓冲
        """
        self._ds_acc = _new_acc()
        self._ds_cls: dict[int, dict] = {}
        self._per_seq: dict[str, dict] = {}
        self._seq_lines: list[str] = []
        self._n_seq = 0
        self._out_dir: Path | None = None
        self._mode = ''
        self._roll_active = False
        self._frames_store: list = []   # [(seq_key, gl, trk_snapshots)] AMOTA 扫描原料 (dataset 级)

    def _reset_seq(self) -> None:
        """
        序列边界重置: 计账/配对表/速度序列/曲线快照清零
        """
        self._acc = _new_acc()
        self._cls_acc: dict[int, dict] = {}
        self._gt_map: dict[int, int] = {}      # gt.id -> 上次配对 trk.id
        self._gt_seen: dict[int, bool] = {}    # gt.id -> 上次出现帧是否配对
        self._pair_cnt: dict = {}              # (seq,label,gt_id,trk_id) -> 共现帧数
        self._gt_frames: dict = {}             # (seq,label,gt_id) -> 出现帧数 (MT/ML 分母)
        self._motion: dict = {}                # (seq,label,gt_id) -> [(fi,vgx,vgy,vdx,vdy)]
        self._curve: list = []                 # 逐帧累计快照 (tp,fp,fn,ids,frag,dist_sum)
        self._fi = 0
        self._seq_key = ''

    # ---- online (eval_mode=1) ----

    def on_seq_start(self, seq_path: str) -> None:
        """
        序列开始: 序列键登记 + 序列级状态清零 (tracker 逐序列调用)
        """
        self._reset_seq()
        self._seq_key = Path(seq_path).name
        self._mode = 'online'

    def online(self, frame, trks: list) -> None:
        """
        在线累计: 逐帧记账 + 单行滚动指标 (MOTA 等可累加项), 全局指标由 on_seq_end 统一算
        """
        self._step(frame.gts, trks)
        if not self.show_panel:
            return
        a = self._acc
        mota = 1 - (a['fp'] + a['fn'] + a['ids']) / a['gt_total'] if a['gt_total'] else float('nan')
        motp = a['dist_sum'] / a['tp'] if a['tp'] else float('nan')
        sys.stdout.write('\r[eval] %s f=%d | tp %d fp %d fn %d ids %d | mota %s motp %s   '
                         % (self._seq_key, self._fi, a['tp'], a['fp'], a['fn'], a['ids'],
                            '-' if mota != mota else '%.1f%%' % (mota * 100),
                            '-' if motp != motp else '%.3f' % motp))
        sys.stdout.flush()
        self._roll_active = True

    def on_seq_end(self, seq_path: str) -> None:
        """
        序列收尾: 本序列指标块 + 曲线落盘 + 并入 dataset 账, 随后清零
        """
        self._finish_seq(seq_path)
        self._reset_seq()

    def on_dataset_end(self) -> dict:
        """
        数据集收尾: 总指标块打印 + 报告落盘, 返回 {per_seq, dataset}
        """
        return self._dataset_end()

    # ---- offline (eval_mode=2) ----

    def evaluate(self, history: list) -> dict:
        """
        离线评估: history[(seq_path, [(gts, trks)])] -> 逐序列记账 + dataset 总计
        """
        self._reset_dataset()                  # 实例复用安全: 入口清数据集级状态
        self._mode = 'offline'
        for seq_path, frames in history:
            self._reset_seq()
            self._seq_key = Path(seq_path).name
            for gts, trks in frames:
                self._step(gts, trks)
            self._finish_seq(seq_path)
        self._reset_seq()
        return self._dataset_end()

    # ---- 收尾共用 ----

    def _finish_seq(self, seq_path: str) -> None:
        """
        单序列收尾: 全局指标折算 + 块打印 + 曲线图 + 标量并入 dataset 账
        """
        seq_name = Path(seq_path).name
        self._accum_global()
        met = self._finalize(self._acc, self._cls_acc)
        lines = self._fmt_block('%s (%d frames)' % (seq_name, self._fi), met)
        if self._roll_active:
            print('')
            self._roll_active = False
        print('\n'.join(lines))
        self._per_seq[seq_name] = met          # 同名序列 (不同目录) 显示层后者覆盖, 指标账不受影响
        self._seq_lines.extend(lines + [''])
        self._n_seq += 1
        if self.do_report:
            self._plot_curve(seq_path)
        _merge_acc(self._ds_acc, self._acc)
        for lb, a in self._cls_acc.items():
            _merge_acc(self._ds_cls.setdefault(lb, _new_acc()), a)

    def _dataset_end(self) -> dict:
        """
        数据集收尾: dataset 账 finalize + AMOTA 扫描 + 块打印 + 报告落盘
        """
        met = self._finalize(self._ds_acc, self._ds_cls)
        met.update(self._calc_amota())            # AMOTA/AMOTP 仅 dataset 级 (阈值扫描需全量帧)
        lines = self._fmt_block('DATASET (%d seqs)' % self._n_seq, met)
        print('\n'.join(lines))
        if self.do_report:
            self._write_report(met)
        return {'per_seq': dict(self._per_seq), 'dataset': met}

    @staticmethod
    def _trk_score(t) -> float:
        """
        航迹置信度: snap_trk 快照读 score, 裸 Trk 读 det_score, 皆无 → 0
        """
        return float(getattr(t, 'score', getattr(t, 'det_score', 0.0)) or 0.0)

    def _calc_amota(self) -> dict:
        """
        AMOTA/AMOTP: det_score 阈值扫描, 每档全量重配对 (贪心+类别约束+门限同 _step), dataset 级 pooled
        """
        store = self._frames_store
        scores = sorted({self._trk_score(t) for _, _, trks in store for t in trks},
                        reverse=True)
        gtotal = sum(len(gl) for _, gl, _ in store)
        if len(scores) < 2 or gtotal == 0:        # 无分数分辨力 (回灌 score=0) 或无 GT → 不定义
            return {'amota': float('nan'), 'amotp': float('nan')}
        if len(scores) > 41:                      # 候选阈值上限: 等距抽稀保扫描耗时
            idx = np.linspace(0, len(scores) - 1, 41).round().astype(int)
            scores = [scores[i] for i in sorted(set(idx.tolist()))]
        pts = []
        for s in scores:
            tp = fp = fn = ids = 0
            dsum = 0.0
            gt_map: dict = {}
            for sq, gl, trks in store:
                sub = [t for t in trks if self._trk_score(t) >= s - 1e-12]
                matched, um_t, um_g = self.match_frame(gl, sub)
                for gt, t, d in matched:
                    tp += 1
                    dsum += d
                    prev = gt_map.get((sq, gt.id))
                    if prev is not None and prev != t.id:
                        ids += 1
                    gt_map[(sq, gt.id)] = t.id
                fp += len(um_t)
                fn += len(um_g)
            recall = tp / gtotal
            pts.append((recall, 1 - (fp + fn + ids) / gtotal,
                        dsum / tp if tp else float('nan'), s))
        picked = []
        for r_t in (i / 10.0 for i in range(10)):  # 每档 recall 目标取最高阈值档 (FP 最少)
            cand = [p for p in pts if p[0] >= r_t - 1e-9]
            if cand:
                picked.append(max(cand, key=lambda x: x[3]))
        if not picked:
            return {'amota': float('nan'), 'amotp': float('nan')}
        motp_v = [p[2] for p in picked if not math.isnan(p[2])]
        return {'amota': float(np.mean([p[1] for p in picked])),
                'amotp': float(np.mean(motp_v)) if motp_v else float('nan')}

    # ---- 指标集合 ----

    def match_frame(self, gts: list, trks: list):
        """
        单帧匹配: 贪心最近邻 + 类别约束 + 距离门限 (向量化, 单测测试缝保留)
        返回 (matched[(gt,trk,dist)], unmatched_trks, unmatched_gts)
        """
        gl = gts.Lst if hasattr(gts, 'Lst') else list(gts)
        tl = list(trks)
        if not gl or not tl:
            return [], tl, gl
        gxy = np.array([[g.x, g.y] for g in gl], dtype=np.float64)
        txy = np.array([[t.x_m, t.y_m] for t in tl], dtype=np.float64)
        glab = np.array([self.gt2label.get(g.type, 0) for g in gl])
        tlab = np.array([int(t.type) for t in tl])
        d = np.linalg.norm(gxy[:, None, :] - txy[None, :, :], axis=2)
        ok = (glab[:, None] == tlab[None, :]) & (glab[:, None] > 0) & (d <= self.match_dist)
        cost = np.where(ok, d, np.inf)
        matched, used_g, used_t = [], np.zeros(len(gl), bool), np.zeros(len(tl), bool)
        if ok.any():
            for f in np.argsort(cost, axis=None):     # 全局按距离升序贪心
                i, j = divmod(int(f), len(tl))
                if not np.isfinite(cost[i, j]):
                    break
                if used_g[i] or used_t[j]:
                    continue
                used_g[i] = used_t[j] = True
                matched.append((gl[i], tl[j], float(d[i, j])))
        unmatched_t = [tl[j] for j in range(len(tl)) if not used_t[j]]
        unmatched_g = [gl[i] for i in range(len(gl)) if not used_g[i]]
        return matched, unmatched_t, unmatched_g

    def _step(self, gts: list, trks: list) -> None:
        """
        单帧记账: 匹配配对 -> tp/ids/frag/配对表/速度序列, 未配对 -> fp/fn
        """
        gl = [g for g in (gts.Lst if hasattr(gts, 'Lst') else gts)
              if not g.isghost and g.type in self.gt2label]   # ghost/未知类不入账
        self._frames_store.append((self._seq_key, gl, list(trks)))   # AMOTA 扫描原料
        matched, um_t, um_g = self.match_frame(gl, trks)
        a = self._acc
        m_ids = set()
        new_map: dict[int, int] = {}            # 帧末统一提交, 防同帧重复 gt.id 即时读污染
        for gt, t, dist in matched:
            lb = int(t.type)
            ca = self._cls_acc.setdefault(lb, _new_acc())
            m_ids.add(gt.id)
            a['tp'] += 1; ca['tp'] += 1
            a['dist_sum'] += dist; ca['dist_sum'] += dist
            prev = self._gt_map.get(gt.id)
            if prev is not None:
                if prev != t.id:                       # ID 切换
                    a['ids'] += 1; ca['ids'] += 1
                if not self._gt_seen.get(gt.id, True):  # 中断后复配
                    a['frag'] += 1; ca['frag'] += 1
            new_map[gt.id] = t.id
            key = (self._seq_key, lb, gt.id, t.id)
            self._pair_cnt[key] = self._pair_cnt.get(key, 0) + 1
            self._motion.setdefault((self._seq_key, lb, gt.id), []).append(
                (self._fi, gt.vx, gt.vy, t.vx_mps, t.vy_mps))
            # 速度误差逐帧账 (幅值无门限, 方向有 |v_gt| 门限)
            vg = math.hypot(gt.vx, gt.vy)
            vd = math.hypot(t.vx_mps, t.vy_mps)
            dvne = abs(vd - vg)
            a['vne_cnt'] += 1; ca['vne_cnt'] += 1
            a['vne_sum'] += dvne; ca['vne_sum'] += dvne
            if vg >= _VEL_GATE:
                da = abs(math.atan2(t.vy_mps, t.vx_mps) - math.atan2(gt.vy, gt.vx))
                da = min(da, 2 * math.pi - da)
                a['vel_cnt'] += 1; ca['vel_cnt'] += 1
                a['vae_sum'] += da; ca['vae_sum'] += da
                if da > math.pi / 2:
                    a['vir_cnt'] += 1; ca['vir_cnt'] += 1
                    a['vaie_sum'] += da; ca['vaie_sum'] += da
        self._gt_map.update(new_map)
        for t in um_t:
            a['fp'] += 1                            # 表外 trk.type 只进 Total 不进类桶 (Total=全部假阳性)
            lb = int(t.type)
            if 1 <= lb <= len(self.class_names):
                self._cls_acc.setdefault(lb, _new_acc())['fp'] += 1
        for gt in um_g:
            a['fn'] += 1
            self._cls_acc.setdefault(self.gt2label[gt.type], _new_acc())['fn'] += 1
        for gt in gl:                               # 本帧各 gt 出现即计 gt_total + 刷新 seen
            lb = self.gt2label[gt.type]
            self._cls_acc.setdefault(lb, _new_acc())['gt_total'] += 1
            gk = (self._seq_key, lb, gt.id)
            self._gt_frames[gk] = self._gt_frames.get(gk, 0) + 1    # MT/ML 分母
            self._gt_seen[gt.id] = gt.id in m_ids
        a['gt_total'] += len(gl)
        if self.do_report:
            self._curve.append((a['tp'], a['fp'], a['fn'], a['ids'], a['frag'], a['dist_sum']))
        self._fi += 1

    def _accum_global(self) -> None:
        """
        全局指标折算 (序列末一次): 配对表 -> idtp/assa_w/MT/ML, 速度序列 -> vse/vde, 累入总量与 per-class 账
        """
        self._accum_one(self._acc, self._pair_cnt, self._motion, self._gt_frames)
        labels = ({k[1] for k in self._pair_cnt} | {k[1] for k in self._motion}
                  | {k[1] for k in self._gt_frames})
        for lb in labels:
            self._accum_one(self._cls_acc.setdefault(lb, _new_acc()),
                            {k: v for k, v in self._pair_cnt.items() if k[1] == lb},
                            {k: v for k, v in self._motion.items() if k[1] == lb},
                            {k: v for k, v in self._gt_frames.items() if k[1] == lb})

    def _accum_one(self, a: dict, pairs: dict, motions: dict, gt_frames: dict) -> None:
        """
        单账折算: 按 seq 分解全局最优分配 (跨序列 id 无连边, 分量不相连, 与联合分配等价) + MT/ML + 速度连续段
        """
        from scipy.optimize import linear_sum_assignment
        from scipy.signal import savgol_filter

        by_seq: dict = {}
        for (sq, _, g, t), c in pairs.items():
            by_seq.setdefault(sq, {})[(g, t)] = c
        for sp in by_seq.values():                # 每序列独立小分配, 免大稠密矩阵
            cnt_g, cnt_t = {}, {}
            for (g, t), c in sp.items():
                cnt_g[g] = cnt_g.get(g, 0) + c
                cnt_t[t] = cnt_t.get(t, 0) + c
            gs, ts = sorted(cnt_g), sorted(cnt_t)
            gi = {g: i for i, g in enumerate(gs)}
            ti = {t: i for i, t in enumerate(ts)}
            W = np.zeros((len(gs), len(ts)))
            for (g, t), c in sp.items():
                W[gi[g], ti[t]] = c
            for r, c in zip(*linear_sum_assignment(-W)):
                tpa = W[r, c]
                if tpa <= 0:
                    continue
                a['idtp'] += tpa
                den = cnt_g[gs[r]] + cnt_t[ts[c]] - tpa    # TPA+FNA+FPA
                if den > 0:
                    a['assa_w'] += tpa * tpa / den         # Σ A(c)·TPA, 帧加权
        matched_n: dict = {}                     # (seq,label,gt_id) -> 被配帧数 (任意 trk)
        for (sq, lb, g, t), c in pairs.items():
            k = (sq, lb, g)
            matched_n[k] = matched_n.get(k, 0) + c
        for k, n in gt_frames.items():            # MT/ML: 单 gt 轨迹被配率 ≥0.8 / ≤0.2
            a['gt_n'] += 1
            r = matched_n.get(k, 0) / n if n > 0 else 0.0
            if r >= 0.8:
                a['mt'] += 1
            elif r <= 0.2:
                a['ml'] += 1
        for series in motions.values():
            for run in _runs(series):
                m = len(run)
                arr = np.asarray(run, dtype=np.float64)
                if m >= 5:                        # VSE: SG 平滑残差 (窗口随段长收缩)
                    w = min(_SG_WIN, m if m % 2 else m - 1)
                    if w >= 5:
                        sm = np.stack([savgol_filter(arr[:, j], w, 2) for j in (3, 4)], 1)
                        a['vse_sum'] += float(np.hypot(sm[:, 0] - arr[:, 3],
                                                       sm[:, 1] - arr[:, 4]).sum())
                        a['vse_cnt'] += m
                if m >= 3:                        # VDE: 速度幅值互相关最优时移
                    sg = np.hypot(arr[:, 1], arr[:, 2])
                    sd = np.hypot(arr[:, 3], arr[:, 4])
                    if sg.std() > 1e-6 and sd.std() > 1e-6:
                        cands = []
                        for lag in range(-_VDE_MAXLAG, _VDE_MAXLAG + 1):
                            u = sd[lag:] if lag >= 0 else sd[:lag]
                            v = sg[:len(sg) - lag] if lag >= 0 else sg[-lag:]
                            if len(u) < 3:
                                continue
                            cc = float(np.corrcoef(u, v)[0, 1])
                            if not math.isnan(cc):
                                cands.append((cc, lag))
                        if cands:
                            # 并列 (浮点噪声级) 时取最小 |lag|, 防线性剖面下时移漂移
                            best_lag = max(cands, key=lambda x: (round(x[0], _CC_TIE),
                                                                 -abs(x[1])))[1]
                            a['vde_sum'] += best_lag * m
                            a['vde_cnt'] += m

    def _finalize(self, acc: dict, cls_acc: dict) -> dict:
        """
        汇总: 账本 (含折算量) -> METRIC_KEYS 指标集合 (total + per-class), 纯除法 O(1)
        """

        def _calc(a: dict) -> dict:
            met = dict.fromkeys(METRIC_KEYS, float('nan'))
            tp, fp, fn = a['tp'], a['fp'], a['fn']
            gt_tot, trk_tot = tp + fn, tp + fp
            met.update(tp=tp, fp=fp, fn=fn, ids=a['ids'], frag=a['frag'])
            met['mota'] = 1 - (fp + fn + a['ids']) / gt_tot if gt_tot else float('nan')
            met['motp'] = a['dist_sum'] / tp if tp else float('nan')
            met['mt'] = a['mt'] / a['gt_n'] if a['gt_n'] else float('nan')
            met['ml'] = a['ml'] / a['gt_n'] if a['gt_n'] else float('nan')
            met['idf1'] = 2 * a['idtp'] / (gt_tot + trk_tot) if (gt_tot + trk_tot) else float('nan')
            met['deta'] = tp / (tp + fp + fn) if (tp + fp + fn) else float('nan')
            met['assa'] = a['assa_w'] / tp if tp else float('nan')
            if tp:
                met['hota'] = math.sqrt(met['deta'] * met['assa'])
                met['loca'] = max(0.0, 1 - met['motp'] / self.match_dist)  # 中心距口径 LocA
            met['vae'] = math.degrees(a['vae_sum'] / a['vel_cnt']) if a['vel_cnt'] else float('nan')
            met['vne'] = a['vne_sum'] / a['vne_cnt'] if a['vne_cnt'] else float('nan')
            met['vaie'] = (math.degrees(a['vaie_sum'] / a['vir_cnt'])
                           if a['vir_cnt'] else (0.0 if a['vel_cnt'] else float('nan')))
            met['vir'] = a['vir_cnt'] / a['vel_cnt'] if a['vel_cnt'] else float('nan')
            met['vse'] = a['vse_sum'] / a['vse_cnt'] if a['vse_cnt'] else float('nan')
            met['vde'] = a['vde_sum'] / a['vde_cnt'] * self.dt if a['vde_cnt'] else float('nan')
            return met

        total = _calc(acc)
        total['classes'] = {lb: _calc(cls_acc[lb]) for lb in sorted(cls_acc)}
        return total

    # ---- 输出 ----

    def _fmt_val(self, k: str, v) -> str:
        """
        指标格式化: nan -> '-'; 百分比/米/度/秒按指标单位
        """
        if isinstance(v, float) and math.isnan(v):
            return '-'
        if k in ('tp', 'fp', 'fn', 'ids', 'frag'):
            return '%d' % v
        if k in ('motp', 'amotp'):
            return '%.3fm' % v
        if k == 'vde':
            return '%+.2fs' % v
        if k in ('mota', 'idf1', 'deta', 'assa', 'hota', 'loca', 'vir', 'mt', 'ml', 'amota'):
            return '%.1f%%' % (v * 100)
        if k in ('vae', 'vaie'):
            return '%.1f°' % v
        return '%.3f' % v

    def _fmt_block(self, tag: str, met: dict) -> list[str]:
        """
        指标块: header + Total 行 + per-class 行, 段落按 METRICS.show 过滤
        """
        def _row(name: str, m: dict) -> str:
            secs = []
            for sec in self._SECTIONS:
                parts = ['%s %s' % (k.upper(), self._fmt_val(k, m[k]))
                         for k in sec if self.show.get(k, 1)]
                if parts:
                    secs.append('  '.join(parts))
            return '  %-10s: %s' % (name, ' | '.join(secs))

        lines = [tag, _row('Total', met)]
        for lb in sorted(met.get('classes', {})):
            if 1 <= lb <= len(self.class_names):
                lines.append(_row(self.class_names[lb - 1], met['classes'][lb]))
        return lines

    def _eval_dir(self) -> Path:
        """
        评估产物目录: 全序列公共父目录/eval.default (与 gt.default 同级惯例)
        """
        if self._out_dir is None:
            paths = [os.path.dirname(str(p)) for p in self.cfg.DATA.paths]
            try:
                root = os.path.commonpath(paths)
            except ValueError:            # 跨盘符等无公共前缀
                root = paths[0]
            self._out_dir = Path(root) / 'eval.default'
        return self._out_dir

    def _plot_curve(self, seq_path: str) -> None:
        """
        指标曲线: 累计 MOTA/MOTP 与 FP/FN/IDS/FRAG 随帧演化 -> PNG
        """
        if len(self._curve) < 2:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        arr = np.asarray(self._curve, dtype=np.float64)
        tp, fp, fn, ids, frag, dsum = arr.T
        x = np.arange(len(tp))
        gt = tp + fn
        mota = np.where(gt > 0, 100 * (1 - (fp + fn + ids) / np.maximum(gt, 1)), np.nan)
        motp = dsum / np.maximum(tp, 1e-9)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        l1, = a1.plot(x, mota, color='tab:blue', label='MOTA [%]')
        a1.set_ylabel('MOTA [%]')
        a1b = a1.twinx()
        l2, = a1b.plot(x, motp, color='tab:orange', label='MOTP [m]')
        a1b.set_ylabel('MOTP [m]')
        a1.legend(handles=[l1, l2], loc='lower right')
        a1.grid(alpha=0.3)
        for k, v, c in (('FP', fp, 'tab:red'), ('FN', fn, 'tab:purple'),
                        ('IDS', ids, 'tab:green'), ('FRAG', frag, 'tab:brown')):
            a2.step(x, v, where='post', color=c, label=k)
        a2.set_xlabel('frame')
        a2.set_ylabel('count')
        a2.legend(loc='upper left')
        a2.grid(alpha=0.3)
        out = self._eval_dir() / ('%s_curve.png' % Path(seq_path).name)
        self._eval_dir().mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, facecolor='white', bbox_inches='tight')
        plt.close(fig)

    def _write_report(self, met: dict) -> None:
        """
        报告落盘: 头部 + 各序列块 + 数据集块 -> eval.default/metrics_<template>.txt
        """
        head = ['=== MOT eval report (%s) ===' % self._mode,
                'seqs: %d  match_dist: %.2fm  dt: %.2fs'
                % (self._n_seq, self.match_dist, self.dt),
                'note: dataset 指标为全帧合并 (micro-pooling), 非 per-seq 平均', '']
        lines = self._fmt_block('DATASET (%d seqs)' % self._n_seq, met)
        out = self._eval_dir() / ('metrics_%s.txt' % self.template)
        self._eval_dir().mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(head + self._seq_lines + lines) + '\n', encoding='utf-8')
        print('  [eval] report -> %s' % out)
