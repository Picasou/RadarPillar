#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 MSRv2 回灌 tracker cfg(单份)。

基于先例 cfg 文本级补丁(保留中文注释):
  RUN.mode → 指定 mode(1=检测回灌 2=跟踪回灌)
  DATA.paths → seq_root 下全部序列子目录(排序)
  MODEL.cfg/ckpt → 本次 MSRv2 训练 resbag 产物
  MANAGER.adapter.type_markov.class_names → 4 类位置表(检测器 label 1-based 位置索引)

用法: python gen_replay_cfg_msrv2.py --base <先例yaml> --mode {1,2} --out <目标yaml>
           --model-cfg <resbag/cfg.yaml> --model-ckpt <resbag/best.pth> --seq-root <拷贝目录>
"""
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--mode', required=True, choices=['1', '2'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-cfg', required=True)
    ap.add_argument('--model-ckpt', required=True)
    ap.add_argument('--seq-root', required=True)
    args = ap.parse_args()

    seqs = sorted(p for p in Path(args.seq_root).iterdir() if p.is_dir())
    if not seqs:
        raise SystemExit(f'[gen_cfg] ERROR: {args.seq_root} 下无序列目录')
    paths_block = '\n'.join('    - %s' % s for s in seqs)

    text = Path(args.base).read_text(encoding='utf-8')

    # RUN.mode (RUN: 段内首个 mode 行)
    text, n = re.subn(r'(RUN:\n# mode: 0=display；1=just_model；2=full\n  mode: )\d', r'\g<1>' + args.mode, text)
    assert n == 1, f'RUN.mode 补丁命中 {n} 处(应为 1)'

    # DATA.paths: 从首个 '  paths:' 行到下一个顶级键之间的路径行整体替换
    pat = re.compile(r'(  paths:\n)(?:    - .*\n)+')
    text, n = pat.subn(r'\g<1>' + paths_block + '\n', text)
    assert n == 1, f'DATA.paths 补丁命中 {n} 处(应为 1)'

    # MODEL.cfg / MODEL.ckpt
    text, n = re.subn(r'(  cfg: ).*', r'\g<1>' + args.model_cfg, text, count=1)
    assert n == 1, 'MODEL.cfg 补丁失败'
    text, n = re.subn(r'(  ckpt: ).*', r'\g<1>' + args.model_ckpt, text, count=1)
    assert n == 1, 'MODEL.ckpt 补丁失败'

    # type_markov.class_names → 5 类位置表 (CLASS_NAMES ['1','2','4','5','7'] ↔ 语义名)
    text, n = re.subn(r'(class_names: )\[.*?\]', r'\g<1>["Car", "Pedestrian", "Cyclist", "Truck", "StaticObject"]', text)
    assert n == 1, f'type_markov.class_names 补丁命中 {n} 处(应为 1)'

    out = Path(args.out)
    out.write_text(text, encoding='utf-8')
    print(f'[gen_cfg] {len(seqs)} 序列, mode={args.mode} → {out}')


if __name__ == '__main__':
    main()
