"""性能评估: MOT 指标 (TP/FP/FN, MOTA/MOTP, IDSW, Frag) — 输出航迹 vs GT, per-class 分账."""
from __future__ import annotations

import numpy as np

# GT 标注类型 → 检测器类名 (AUTOSIL GTT 预设: 1=轿车 2=行人 4=二轮车 5=卡车 7=静态障碍物)
GT_TYPE_NAMES = {1: 'Car', 2: 'Pedestrian', 4: 'Cyclist', 5: 'Truck', 7: 'StaticObject'}


class Evaluator:
    """
    in : online→(FRAME, 本帧输出航迹) / on_seq_end→序列名 / evaluate→history[seq[(gts, trks)]]
    out: MOT 指标 (TP/FP/FN/MOTA/MOTP/IDSW/Frag, per-class) 控制台 + 报告文件
    """

    def __init__(self, cfg, class_names: list[str] | None = None) -> None:
        self.cfg = cfg
        self.class_names = list(class_names or ['Car', 'Pedestrian', 'Cyclist'])
        self.match_dist = float(getattr(cfg.EVALUATE, 'match_dist', 2.0))
        self.do_report = int(getattr(cfg.EVALUATE, 'report', 0)) == 1
        self.show = dict(getattr(cfg.METRICS, 'show', {}) or {})
        # GT 类型 → 检测 label (1-based)。两种对齐: 类名为数字串时按数值 (MSR: 枚举即类名),
        # 否则按名字 (Car/Pedestrian/Cyclist)。检测器无对应类 → 不入表 (评估排除)
        self.gt2label = {}
        for i, name in enumerate(self.class_names):
            if name.strip().isdigit():
                t = int(name)
                self.gt2label[t] = i + 1
            else:
                for gt_t, n in GT_TYPE_NAMES.items():
                    if n == name:
                        self.gt2label[gt_t] = i + 1
        self._reset_acc()

    # ---- 累计器 ----

    @staticmethod
    def _new_acc() -> dict:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'idsw': 0, 'frag': 0,
                'dist_sum': 0.0, 'gt_total': 0}

    def _reset_acc(self) -> None:
        self._acc = self._new_acc()
        self._cls_acc: dict[int, dict] = {}
        self._gt_map: dict[int, int] = {}    # gt.id -> 上次配对 trk.id
        self._gt_seen: dict[int, bool] = {}  # gt.id -> 上帧是否配对 (Frag 判定)

    # ---- 单帧匹配 ----

    def match_frame(self, gts: list, trks: list):
        """
        单帧匹配: 贪心最近邻 + 类别约束 + 距离门限
        返回 (matched[(gt,trk,dist)], unmatched_trks, unmatched_gts); 无对应检测类的 GT 整体剔除
        """
        gts_v = [g for g in gts if g.type in self.gt2label]
        pairs = []
        for g in gts_v:
            label = self.gt2label[g.type]
            for t in trks:
                if t.type != label:
                    continue
                d = float(np.hypot(g.x - t.x_m, g.y - t.y_m))
                if d <= self.match_dist:
                    pairs.append((d, g, t))
        pairs.sort(key=lambda p: p[0])
        used_g, used_t, matched = set(), set(), []
        for d, g, t in pairs:
            if id(g) in used_g or id(t) in used_t:
                continue
            used_g.add(id(g))
            used_t.add(id(t))
            matched.append((g, t, d))
        unmatched_trks = [t for t in trks if id(t) not in used_t]
        unmatched_gts = [g for g in gts_v if id(g) not in used_g]
        return matched, unmatched_trks, unmatched_gts

    def _step(self, gts: list, trks: list) -> None:
        """
        单帧累计: TP/FP/FN + IDSW(复配换号) + Frag(断后重接), 总账与 per-class 同步
        """
        acc, cls_acc = self._acc, self._cls_acc
        matched, fp_trks, fn_gts = self.match_frame(gts, trks)
        acc['tp'] += len(matched)
        acc['fp'] += len(fp_trks)
        acc['fn'] += len(fn_gts)
        acc['gt_total'] += len(matched) + len(fn_gts)
        acc['dist_sum'] += sum(d for _, _, d in matched)
        for g, t, d in matched:
            c = cls_acc.setdefault(self.gt2label[g.type], self._new_acc())
            c['tp'] += 1
            c['gt_total'] += 1
            c['dist_sum'] += d
            prev = self._gt_map.get(g.id)
            if prev is not None:
                if prev != t.id:
                    acc['idsw'] += 1
                    c['idsw'] += 1
                if self._gt_seen.get(g.id) is False:
                    acc['frag'] += 1
                    c['frag'] += 1
            self._gt_map[g.id] = t.id
            self._gt_seen[g.id] = True
        for g in fn_gts:
            self._gt_seen[g.id] = False
            c = cls_acc.setdefault(self.gt2label[g.type], self._new_acc())
            c['fn'] += 1
            c['gt_total'] += 1
        for t in fp_trks:
            c = cls_acc.setdefault(int(t.type), self._new_acc())
            c['fp'] += 1

    # ---- online (eval_mode=1) ----

    def online(self, frame, trks: list) -> None:
        """
        在线累计: 逐帧记账 (配对结果由 on_seq_end 打印)
        """
        self._step(frame.gts.Lst, trks)

    def on_seq_end(self, seq_name: str) -> None:
        """
        序列收尾: 打印本序列 online 累计并清零 (跨序列状态不延续)
        """
        a = self._finalize(self._acc)
        print('  [eval:%s] TP %d FP %d FN %d IDSW %d Frag %d MOTA %.3f MOTP %.2fm'
              % (seq_name, a['tp'], a['fp'], a['fn'], a['idsw'], a['frag'],
                 a['mota'], a['motp']))
        self._reset_acc()

    # ---- offline (eval_mode=2) ----

    def evaluate(self, history: list) -> dict:
        """
        离线评估: history[seq[(gts, trks)]] -> per-seq/per-class + 总计, 打印 + 报告落盘
        """
        total = self._new_acc()
        total_cls: dict[int, dict] = {}
        lines, seqs = [], []
        for si, seq in enumerate(history):
            if not seq or all(not g.num for g, _ in seq):
                lines.append('seq %d: 无 GT, 跳过' % si)
                continue
            self._reset_acc()
            for gts, trks in seq:
                self._step(gts.Lst, trks)
            a = self._finalize(self._acc)
            seqs.append(a)
            lines.append(self._fmt('seq %d' % si, a))
            for label, c in self._cls_acc.items():
                lines.append(self._fmt('  %s' % self._label_name(label),
                                       self._finalize(c), indent='    '))
                self._merge(total_cls.setdefault(label, self._new_acc()), c)
            self._merge(total, self._acc)
        total = self._finalize(total)
        lines.append(self._fmt('TOTAL', total))
        for label, c in sorted(total_cls.items()):
            lines.append(self._fmt('  %s' % self._label_name(label),
                                   self._finalize(c), indent='  '))
        text = '\n'.join(lines)
        print('\n  [eval:offline]\n' + text)
        if self.do_report:
            import os
            os.makedirs('output/tracker_eval', exist_ok=True)
            with open('output/tracker_eval/report.txt', 'a', encoding='utf-8') as f:
                f.write(text + '\n\n')
        return {'total': total, 'per_class': total_cls, 'seqs': seqs}

    # ---- 工具 ----

    def _merge(self, dst: dict, src: dict) -> None:
        for k, v in src.items():
            dst[k] += v

    def _finalize(self, a: dict) -> dict:
        out = dict(a)
        out['mota'] = 1.0 - (a['fn'] + a['fp'] + a['idsw']) / a['gt_total'] if a['gt_total'] else 0.0
        out['motp'] = a['dist_sum'] / a['tp'] if a['tp'] else 0.0
        return out

    def _want(self, key: str) -> bool:
        return int(self.show.get(key, 1)) == 1

    def _fmt(self, name: str, a: dict, indent: str = '  ') -> str:
        parts = [name + ':']
        for key, label in (('tp', 'TP'), ('fp', 'FP'), ('fn', 'FN'),
                           ('idsw', 'IDSW'), ('frag', 'Frag'),
                           ('mota', 'MOTA'), ('motp', 'MOTP')):
            if not self._want('ids' if key == 'idsw' else key):
                continue
            val = '%.3f' % a[key] if key in ('mota',) else (
                '%.2fm' % a[key] if key == 'motp' else str(a[key]))
            parts.append('%s=%s' % (label, val))
        return indent + ' '.join(parts)

    def _label_name(self, label: int) -> str:
        return self.class_names[label - 1] if 1 <= label <= len(self.class_names) else str(label)
