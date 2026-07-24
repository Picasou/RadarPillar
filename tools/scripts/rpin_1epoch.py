#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 前置 Task 4：全 22 cfg 1-epoch 验证（NaN/OOM 自愈 wrapper，plan §0.5 S12/S13）。

串行单卡跑；OOM 降 bs 序列 16→8→4→2→1（重档从 8 起）；结构性 NaN 预筛后 BLOCKED。
结果写到 .tmp/rpin_prereq/1epoch_results.json + stdout 摘要。

用法: python tools/scripts/rpin_1epoch.py [--cfgs a0,a2,...] [--bs-default 16] [--gpu 0]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path('.')
CST = timezone(timedelta(hours=8))
YAML_DIR = REPO / 'experiments' / 'YAML'
LOG_DIR = REPO / 'output' / 'train_log' / 'vod'
LOG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = REPO / '.tmp' / '2026-07-23' / 'rpin_prereq'
RES_DIR.mkdir(parents=True, exist_ok=True)

# 与 derive_rpin_sh.py 一致：bs S11
DEFAULT_BS = {
    'a0': 16, 'a1': 16, 'a2': 16, 'a3': 16,
    'b1': 16, 'b2': 16, 'b3': 8, 'b4': 8,
    'n1': 16, 'n2': 16, 'n3': 8, 'n4': 16, 'n5': 8, 'n6': 8,
    'head_anchor': 16, 'head_center': 8, 'head_2d': 8,
    'e1': 16, 'e2': 16, 'e3': 16, 'f1': 16, 'f3': 16,
}
# BLOCKED 允许（敏感档）：b3/b4/n3/n5/n6/f1
BLOCKED_ALLOWED = {'b3', 'b4', 'n3', 'n5', 'n6', 'f1'}
# 核心必过：a0/a2/a3 + b1/b2 + n1/n2/n4 + head_anchor/head_center/head_2d + e1/e3
CORE_REQUIRED = {'a0', 'a2', 'a3', 'b1', 'b2', 'n1', 'n2', 'n4',
                  'head_anchor', 'head_center', 'head_2d', 'e1', 'e3'}


def parse():
    ap = argparse.ArgumentParser(description='RPiN 1-epoch 验证调度器')
    ap.add_argument('--cfgs', default=None, help='逗号分隔 cfg tag；缺省=全部 22')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs-default', type=int, default=16)
    return ap.parse_args()


def run_one(tag: str, gpu: int) -> dict:
    """对单个 cfg 跑 1-epoch，按 OOM 自愈降 bs。返回 {tag, status, attempts, final_bs, msg}。"""
    cfg = YAML_DIR / f'{tag}.yaml'
    if not cfg.exists():
        return {'tag': tag, 'status': 'MISSING', 'attempts': 0, 'final_bs': None, 'msg': 'cfg not found'}

    start_bs = DEFAULT_BS.get(tag, 16)
    bs_seq = [start_bs]
    for lower in (8, 4, 2, 1):
        if lower < start_bs:
            bs_seq.append(lower)
    attempts = []
    ts = datetime.now(CST).strftime('%Y%m%d%H%M%S')
    for attempt, bs in enumerate(bs_seq, 1):
        out_root = LOG_DIR / f'{ts}_rpin1ep_{tag}_bs{bs}'
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            'python', '-u', 'tools/train.py',
            '--cfg_file', str(cfg),
            '--batch_size', str(bs),
            '--workers', '2',
            '--epochs', '1',
            '--extra_tag', f'rpin1ep_{tag}',
            '--output_root', str(out_root),
            '--skip_eval',
            '--set', 'OPTIMIZATION.early_stop.enabled', 'False',
            'OPTIMIZATION.LR_WARMUP', 'False',
        ]
        # 机器无关（plan 自审 + §0.5 S4）：继承当前激活环境（base），仅覆盖可见 GPU。
        # 旧版硬编码 /home/dministrator1/.../angle 死路径 → 子进程 PYTHONHOME 失效，必崩。
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
            dt = time.time() - t0
        except subprocess.TimeoutExpired:
            attempts.append({'bs': bs, 'dt': 900, 'exit': 'TIMEOUT'})
            continue
        stderr_full = proc.stderr or ''
        stderr_tail = stderr_full[-400:]
        if proc.returncode == 0:
            # 检查 NaN：抓 train log 中 "loss=nan" 或 "nan" 标记
            # H5 修复：NaN 信号改从可靠源取。train.py 实际写 log_train_*.txt（非 train.log），
            # 逐 iter loss 只进 tqdm(stderr)+tensorboard。旧版读 train.log（永不存在）→回退
            # stdout（无 loss）→正则永假→NaN-loss(rc=0) 误报 OK。现扫 stderr ∪ log_train_*.txt
            # 判 loss 是否 nan，并校验 ckpt 落盘，S13 结构性 NaN 预筛据此真正生效。
            log_files = sorted(out_root.glob('log_train_*.txt'))
            log_path = str(log_files[-1]) if log_files else ''
            log_text = stderr_full
            if log_files:
                log_text = log_files[-1].read_text(encoding='utf-8', errors='ignore') + '\n' + log_text
            nan = bool(re.search(r'loss\s*[=:]\s*nan|NaN detected', log_text, re.I))
            ckpt_ok = bool(list(out_root.rglob('checkpoint_epoch_1.pth')))
            if nan:
                attempts.append({'bs': bs, 'dt': round(dt, 1), 'exit': 'NAN'})
                # 结构性 NaN 预筛：第一次就 NaN → 跳 bs 循环
                if attempt == 1:
                    return {'tag': tag, 'status': 'BLOCKED_NAN', 'attempts': attempts,
                            'final_bs': bs, 'msg': '结构性 NaN（首步即 nan）'}
                continue
            if not ckpt_ok:
                return {'tag': tag, 'status': 'FAIL', 'attempts': attempts,
                        'final_bs': bs, 'msg': 'rc=0 但未落 checkpoint_epoch_1.pth'}
            return {'tag': tag, 'status': 'OK', 'attempts': attempts,
                    'final_bs': bs, 'msg': f'{dt:.1f}s bs={bs}', 'log': log_path}
        # 退出非 0：判 OOM（显存/资源）
        if 'CUDA out of memory' in proc.stderr or 'OutOfMemory' in proc.stderr:
            attempts.append({'bs': bs, 'dt': round(dt, 1), 'exit': 'OOM'})
            continue
        attempts.append({'bs': bs, 'dt': round(dt, 1), 'exit': f'RC={proc.returncode}'})
        # 非 OOM 错误：直接记 FAIL，不降 bs
        return {'tag': tag, 'status': 'FAIL', 'attempts': attempts,
                'final_bs': bs,
                'msg': (stderr_tail[-300:] if stderr_tail else proc.stdout[-300:])}
    # M6：耗尽 bs 序列后按最后一次 attempt 的退出类型分类（不再硬编码 BLOCKED_OOM）
    last_exit = attempts[-1]['exit'] if attempts else 'OOM'
    if last_exit == 'NAN':
        return {'tag': tag, 'status': 'BLOCKED_NAN', 'attempts': attempts,
                'final_bs': bs_seq[-1], 'msg': f'bs 降到 {bs_seq[-1]} 仍 NaN'}
    return {'tag': tag, 'status': 'BLOCKED_OOM', 'attempts': attempts,
            'final_bs': bs_seq[-1], 'msg': f'bs 降到 {bs_seq[-1]} 仍 OOM'}


def main():
    args = parse()
    tags = args.cfgs.split(',') if args.cfgs else list(DEFAULT_BS.keys())
    print(f'[1epoch] {len(tags)} 个 cfg，gpu={args.gpu}', flush=True)
    results = []
    t_start = time.time()
    for i, tag in enumerate(tags, 1):
        print(f'[{i}/{len(tags)}] {tag} ...', flush=True)
        r = run_one(tag, args.gpu)
        results.append(r)
        print(f'  → {r["status"]} | bs={r.get("final_bs")} | {r.get("msg", "")}', flush=True)
    dt_total = time.time() - t_start
    # 保存结果
    out_json = RES_DIR / '1epoch_results.json'
    out_json.write_text(json.dumps({
        'time_total_s': round(dt_total, 1),
        'n': len(results),
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    # 摘要
    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_blocked = sum(1 for r in results if r['status'].startswith('BLOCKED'))
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')
    print(f'\n=== SUMMARY ===')
    print(f'  OK={n_ok}  BLOCKED={n_blocked}  FAIL={n_fail}  time={dt_total:.1f}s')
    core_results = {r['tag']: r['status'] for r in results if r['tag'] in CORE_REQUIRED}
    core_fail = [t for t, s in core_results.items() if s != 'OK']
    print(f'  Core (must pass): {len(core_results) - len(core_fail)}/{len(core_results)} OK'
          + (f'  fail={core_fail}' if core_fail else ''))
    # 核心必过任一未过 → 退出 1
    if core_fail:
        print('FAIL: 核心 cfg 未全过')
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
