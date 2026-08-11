#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_conclusion.py — 聚合 resbag 的 model_store.yaml, 生成跨实验结论文档 (md)

整个 pipeline (所有模型) 跑完后由 workflow 调用一次。
源: <root>/<dataset>/*/model_store.yaml (resbag 产物)
出: <root>/<dataset>/_conclusion.md (对比表 + 最优 + 异常)

用法:
  python make_conclusion.py --dataset vod [--root output/train_log] [-o out.md]
"""
import argparse
import sys
from pathlib import Path

import yaml


def fmt(v, spec='{:.2f}'):
    if v is None:
        return '—'
    try:
        return spec.format(float(v))
    except (ValueError, TypeError):
        return str(v)


def main():
    ap = argparse.ArgumentParser(description='聚合 model_store.yaml 生成跨实验结论')
    ap.add_argument('--dataset', required=True, help='数据集名 (决定 glob 父目录)')
    ap.add_argument('--root', default='output/train_log', help='train_log 根')
    ap.add_argument('-o', '--out', default=None, help='输出 md (默认 <root>/<dataset>/_conclusion.md)')
    args = ap.parse_args()

    root = Path(args.root) / args.dataset
    stores = sorted(root.glob('*/model_store.yaml'))
    if not stores:
        sys.exit(f'[conclusion] 无 model_store.yaml 于 {root}（resbag 是否跑过？）')

    rows = []
    for s in stores:
        try:
            d = yaml.safe_load(s.read_text(encoding='utf-8')) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f'[conclusion] 跳过 {s}: {e}', file=sys.stderr)
            continue
        r40 = d.get('map_r40') or {}
        rows.append({
            'folder': d.get('folder', s.parent.name),
            'tag': d.get('tag', ''),
            'mean': r40.get('mean'),
            'car': r40.get('car'),
            'ped': r40.get('pedestrian'),
            'cyc': r40.get('cyclist'),
            'params': d.get('params_m'),
            'flops': d.get('flops_g'),
            'status': d.get('status', ''),
            'note': d.get('note', ''),
        })

    lines = []
    lines.append(f'# {args.dataset} 跨实验结论\n')
    lines.append(f'聚合 {len(rows)} 个实验 (源: `{root}/*/model_store.yaml`)\n')
    lines.append('## 核心结果对比 (map@R40 moderate, EAA)\n')
    lines.append('| 实验 | tag | mAP | Car | Ped | Cyclist | Params(M) | FLOPS(G) | status |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda x: (x['mean'] is None, -(x['mean'] or 0))):
        lines.append(
            f"| {r['folder']} | {r['tag']} | {fmt(r['mean'])} | {fmt(r['car'])} | "
            f"{fmt(r['ped'])} | {fmt(r['cyc'])} | {fmt(r['params'])} | {fmt(r['flops'])} | {r['status']} |"
        )

    done = [r for r in rows if r['mean'] is not None]
    if done:
        best = max(done, key=lambda x: x['mean'])
        lines.append('\n## 最优 (按 mAP mean)\n')
        lines.append(f"**{best['folder']}** (tag={best['tag']}): mAP={fmt(best['mean'])}, "
                     f"params={fmt(best['params'])}M, flops={fmt(best['flops'])}G")

    partial = [r for r in rows if r['status'] not in ('done', '')]
    if partial:
        lines.append('\n## 需关注 (status 非 done)\n')
        for r in partial:
            lines.append(f"- {r['folder']}: status={r['status']} note={r['note']}")

    out = Path(args.out) if args.out else root / '_conclusion.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[conclusion] 写入 {out} ({len(rows)} 实验)')


if __name__ == '__main__':
    main()
