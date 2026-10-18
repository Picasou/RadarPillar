"""
逐帧回灌日志: mode2 下每序列一份文件, 序列头关键配置 + 逐帧 PRE/DET/MATCH/NEW/COAST/DEAD/OUT。
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path


class FrameLogger:
    """
    逐帧日志落盘: 跟踪链各阶段快照 → 文本块 (埋点在 Tracker.tracker_step, 事件归因零侵入 diff)。
    """

    def __init__(self, cfg, manager, relpath: str = 'radar.default') -> None:
        # 参数集中: 事件归因与表头直接读权威源 (manager 实例 > cfg)
        self.enabled = (cfg.RUN.mode == 2 and cfg.LOG.enable == 1)
        self.level = cfg.LOG.level
        self.dt = cfg.MANAGER.dt
        self._relpath = relpath   # 数据流子目录 (对齐 loader.relpath, 0201 同级)
        self.manager = manager
        tm = cfg.MANAGER.adapter.get('type_markov', {})
        self.class_names = tm.get('class_names', [])
        self._model_cfg = cfg.MODEL.cfg
        # 模型名取 ckpt 路径的训练目录段 (.../<训练folder>/resbag/best.pth), 比 cfg 文件名更可读
        ck = Path(cfg.MODEL.ckpt)
        self._ckpt = cfg.MODEL.ckpt
        self._model_name = ck.parent.parent.name if ck.parent.name == 'resbag' else ck.parent.name
        self._ftr = (cfg.FILTER.type, cfg.MATCH.gap_type, cfg.MATCH.gap_dim, cfg.MATCH.thresh,
                     cfg.MANAGER.birth_heat, cfg.MANAGER.death_heat, cfg.MANAGER.merge_dist,
                     cfg.MANAGER.birth_min_range, cfg.MODEL.score_thresh)
        # dir 空=日志随数据走 (写 <序列>/frame_log_mode2.log); 非空=集中存放 <dir>/<时间戳_run>/
        self._dir = cfg.LOG.dir.strip()
        self._fh = None
        self._seq_frames = 0

    # ---------- 类型枚举 → 语义名 (obj.type/trk.type 为 1-based 位置索引) ----------
    def _cls(self, t: int) -> str:
        """
        类型语义化: 1-based 枚举 → class_names 表项, 越界回退原值。
        """
        return self.class_names[t - 1] if 1 <= t <= len(self.class_names) else str(t)

    # ---------- 序列生命周期 ----------
    def begin_seq(self, seq_path: str, n_frames: int) -> None:
        """
        序列开表: 建 <dir>/<时间戳_run>/<序列名>.log 并写关键配置头。
        """
        if not self.enabled:
            return
        if self._dir:
            self._root = Path(self._dir) / (datetime.now().strftime('%Y%m%d-%H%M%S') + '_run')
            self._root.mkdir(parents=True, exist_ok=True)
            out = self._root / (Path(seq_path).name + '.log')
        else:
            # 随数据目录: <序列>/<relpath>/frame_log_mode2.log (0201 同目录同批同源)
            out = Path(seq_path) / self._relpath / 'frame_log_mode2.log'
        self._fh = open(out, 'w', encoding='utf-8')
        self._seq_frames = 0
        ftype, gap_type, gap_dim, thresh, bh, dh, md, bmr, st = self._ftr
        fname = {1: 'abf', 2: 'KF', 3: 'EKF', 4: 'IMM'}.get(ftype, str(ftype))
        gname = {1: 'euc', 2: 'mah'}.get(gap_type, str(gap_type))
        self._fh.write(f"================ SEQ {Path(seq_path).name} ================\n")
        self._fh.write(f"[SEQ-CFG] model={self._model_name}  ckpt={self._ckpt}\n")
        self._fh.write(f"[SEQ-CFG] filter={fname}  match={gname}_xy{'+' if gap_dim == 3 else ''}"
                       f" thresh={thresh}m  dt={self.dt}s\n")
        self._fh.write(f"[SEQ-CFG] birth_heat={bh}  death_heat={dh}  merge_dist={md}m"
                       f"  birth_min_range={bmr}m  score_thresh={st}\n")
        self._fh.write(f"[SEQ-CFG] frames={n_frames}\n\n")

    def end_seq(self) -> None:
        """
        序列收表: 关闭句柄。
        """
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ---------- 帧内快照 (埋点处按阶段调用, 状态随时序变化必须当场取值) ----------
    def snap_pre(self, trks: list) -> list:
        """
        predict 前快照: 逐 trk 取值元组 (predict 原地改状态, 引用无效)。
        """
        return [(t.id, t.lifetime_s, t.x_m, t.y_m, t.vx_mps, t.vy_mps, t.heading_deg,
                 t.length_m, t.width_m, t.height_m, t.type, t.type_confi,
                 t.obstacle_prob, t.measurement_status, t.existence_prob) for t in trks]

    def snap_matches(self, matches, objs: list) -> list:
        """
        matcher 后快照: matched 对 (tid, obj帧内号, gap), 须在 updater 改状态前算 (此时 trk=预测态)。
        """
        idx = {id(o): i for i, o in enumerate(objs)}
        return [(t.id, idx.get(id(o), -1), math.hypot(t.x_m - o.x, t.y_m - o.y))
                for t, o in matches.matched]

    # ---------- 整帧落盘 ----------
    def log_frame(self, idx: int, pre_snaps: list, objs: list, match_pairs: list,
                  unmatched_objs: list, unmatched_trks: list, pre_ids: set, post_trks: list) -> None:
        """
        整帧写块: manager.run 后一次成文 (new/dead 由 id diff 归因, dead 原因用 pre 快照重算)。
        """
        if not self.enabled or self._fh is None:
            return
        self._seq_frames += 1
        post_ids = {t.id for t in post_trks}
        new_ids = post_ids - pre_ids
        dead_ids = pre_ids - post_ids
        snap_map = {s[0]: s for s in pre_snaps}
        new_trks = [t for t in post_trks if t.id in new_ids]
        out_trks = [t for t in post_trks if t.obstacle_prob == 1]
        coast_ids = {t.id for t in unmatched_trks}

        # obj 帧内号: unmatched_objs 是 objs 子集, 按 id() 映射回帧内编号
        obj_no = {id(o): i for i, o in enumerate(objs)}
        # 新建轨回配源 obj: 初建 x/y 精确拷贝自 obj (manager._man_create_utils), 全等配对
        src = {}
        used_o = set()
        for t in new_trks:
            for i, o in enumerate(objs):
                if i in used_o:
                    continue
                if o.x == t.x_m and o.y == t.y_m and o.z == t.z_m:
                    src[t.id] = i
                    used_o.add(i)
                    break
        dropped = [(obj_no[id(o)], o) for o in unmatched_objs
                   if obj_no.get(id(o), -1) not in src.values() and obj_no.get(id(o), -1) not in used_o]

        n_m, n_new, n_coast, n_dead = len(match_pairs), len(new_trks), len(coast_ids & post_ids), len(dead_ids)

        if self.level == 'slim':
            self._fh.write(f"frame {idx:04d} (t={idx * self.dt:.2f}s): "
                           f"pre={len(pre_snaps)} det={len(objs)} match={n_m} new={n_new} "
                           f"coast={n_coast} dead={n_dead} out={len(out_trks)}\n")
            return

        # ---------- full ----------
        # 列宽约定(等宽字体纵向成列): id=<3d 标签统一"  xxx#NNN "前缀; pos=8.2f vel=7.2f hdg=6.1f
        # dop=6.2f score=4.2f gap=5.2f size=4.1f type=<13s(+3d) —— 含负号与量程余量, 全段共用
        self._fh.write(f"---- frame {idx:04d} (t={idx * self.dt:6.2f}s) ----\n")
        self._fh.write(f"[PRE] trks={len(pre_snaps)} (pre-predict)\n")
        for s in pre_snaps:
            tid, age, x, y, vx, vy, hdg, ln, w, h, tp, tc, obst, streak, pex = s
            self._fh.write(f"  trk#{tid:<3d} age={age:6.2f}s pos=({x:8.2f},{y:8.2f}) "
                           f"vel=({vx:7.2f},{vy:7.2f}) hdg={hdg:6.1f}° size=({ln:4.1f},{w:4.1f},{h:4.1f}) "
                           f"type={self._cls(tp):<13s}({tc:3d}) obst={obst:d} streak={streak:2d} P_ex={pex:3d}\n")
        self._fh.write(f"[DET] objs={len(objs)}\n")
        for i, o in enumerate(objs):
            self._fh.write(f"  obj#{i:<3d} score={o.score:4.2f} pos=({o.x:8.2f},{o.y:8.2f},z={o.z:5.2f}) "
                           f"vel=({o.vx:7.2f},{o.vy:7.2f}) dop={o.doppler:6.2f} hdg={o.heading:6.1f}° "
                           f"size=({o.length:4.1f},{o.width:4.1f},{o.height:4.1f}) type={self._cls(o.type):<13s}\n")
        self._fh.write(f"[MATCH] matched={n_m}  new={n_new}  coasting={n_coast}  dead={n_dead}\n")
        for tid, oi, gap in match_pairs:
            self._fh.write(f"  trk#{tid:<3d} <-> obj#{oi:<3d} gap={gap:5.2f}m\n")
        # coasting 单独一栏: 未关联且仍存活的 trk (断粮滑行, streak 递增)
        for t in unmatched_trks:
            if t.id in coast_ids & post_ids:
                s = snap_map.get(t.id)
                streak = t.measurement_status if s is None else s[13] + 1
                self._fh.write(f"[COAST] trk#{t.id:<3d} pos=({t.x_m:8.2f},{t.y_m:8.2f}) "
                               f"vel=({t.vx_mps:7.2f},{t.vy_mps:7.2f}) type={self._cls(t.type):<13s}"
                               f"obst={t.obstacle_prob:d} streak={streak:d}/{self.manager.death_heat:d}\n")
        for t in new_trks:
            oi = src.get(t.id, -1)
            self._fh.write(f"[NEW] obj#{oi:<3d} -> trk#{t.id:<3d}\n")
        for oi, o in dropped:
            if math.hypot(o.x, o.y) < self.manager.birth_min_range:
                why = 'near_ego<birth_min_range'
            elif any(math.hypot(o.x - t.x_m, o.y - t.y_m)
                     < min(0.5 * math.hypot(o.length, o.width),
                           0.5 * math.hypot(t.length_m, t.width_m)) for t in post_trks):
                why = 'near_existing_trk'
            else:
                why = 'id_pool_full/other'
            self._fh.write(f"[DROP] obj#{oi:<3d} 未建轨 ({why})\n")
        for tid in sorted(dead_ids):
            s = snap_map.get(tid)
            if s is not None:
                _, _, x, y, _, _, _, _, _, _, _, _, _, streak, _ = s
                if streak + 1 >= self.manager.death_heat:
                    why = f'meas_streak={streak + 1} >= death_heat={self.manager.death_heat}'
                elif self.manager.bound is not None:
                    x_lo, x_hi, y_lo, y_hi = self.manager.bound
                    why = '出界' if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi) else 'merged'
                else:
                    why = 'merged'
            else:
                why = 'unknown'
            self._fh.write(f"[DEAD] trk#{tid:<3d} ({why})\n")
        self._fh.write(f"[OUT] trks={len(out_trks)}\n")
        for t in out_trks:
            self._fh.write(f"  trk#{t.id:<3d} pos=({t.x_m:8.2f},{t.y_m:8.2f}) "
                           f"vel=({t.vx_mps:7.2f},{t.vy_mps:7.2f}) hdg={t.heading_deg:6.1f}° "
                           f"type={self._cls(t.type):<13s}({t.type_confi:3d}) obst={t.obstacle_prob:d} P_ex={t.existence_prob:3d}\n")
        self._fh.write('\n')
