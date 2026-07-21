#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长周期任务阶段存档（checkpoint snapshot）。

语义类似 git commit：每完成一个阶段打存档点，崩溃后从最近存档恢复。
但【不进 git】——模型 .pth 等大文件由训练脚本自己存到 output/，本脚本只记
「产物在什么位置 + 数据结果是什么」的元数据指针。

子命令：
  save         打一个阶段存档点（记产物位置 + 结果 + 下一阶段起点）
  list         列任务所有存档点
  show-latest  读最近存档（长任务启动恢复用）

设计原则：纯元数据索引，绝不复制大文件。确定性，不依赖大模型。

目录约定：存档索引在 .tmp/<YYYY-MM-DD>/<slug>/<slug>_checkpoints.json
日期取自 init_lt_task.py 写入的 task_meta.json（任务启动日，跨天不挪窝）。
找不到 meta 时 fallback 当日。
"""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = ROOT / '.tmp'


def _today() -> str:
    return datetime.now(CST).strftime('%Y-%m-%d')


def _resolve_task_dir(task_slug: str) -> tuple[Path, bool]:
    """定位任务目录：先 rglob 找 task_meta.json（启动日），找不到则 fallback 当日。

    Returns: (task_dir, from_meta)
    """
    if TMP_DIR.exists():
        candidates = []
        for meta in TMP_DIR.glob(f'*/{task_slug}/task_meta.json'):
            try:
                m = json.loads(meta.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                continue
            if m.get('slug') != task_slug:
                continue
            try:
                mtime = meta.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, Path(meta).parent))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1], True
    return TMP_DIR / _today() / task_slug, False


def _archive_path(task_slug: str) -> tuple[Path, bool]:
    task_dir, from_meta = _resolve_task_dir(task_slug)
    return task_dir / f'{task_slug}_checkpoints.json', from_meta


def _load(task_slug: str) -> list:
    p, _ = _archive_path(task_slug)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []


def _save(task_slug: str, data: list) -> Path:
    p, from_meta = _archive_path(task_slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return p


# ════════════════════════════════════════════════════════════════
#  save: 打存档点
# ════════════════════════════════════════════════════════════════
def cmd_save(args):
    """产物位置/结果都作为自由键值对传入，脚本只负责落盘归档，不解析大文件。"""
    artifacts = {}
    for kv in args.artifact or []:
        if '=' in kv:
            k, v = kv.split('=', 1)
            artifacts[k.strip()] = v.strip()
    results = {}
    for kv in args.result or []:
        if '=' in kv:
            k, v = kv.split('=', 1)
            results[k.strip()] = v.strip()

    data = _load(args.task)
    stage_idx = len(data) + 1
    record = {
        'stage_idx': stage_idx,
        'stage': args.stage,
        'time': datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST'),
        'artifacts': artifacts,        # 产物位置指针（ckpt/best/loss图/eval 路径）
        'results': results,            # 数据结果（metric/loss 摘要）
        'next_start': args.next_start, # 下一阶段起点（崩溃恢复从这续）
        'note': args.note or '',
    }
    data.append(record)
    p = _save(args.task, data)
    print(f'[checkpoint] 已打存档点 #{stage_idx}: {args.stage}')
    print(f'[checkpoint] 存档索引: {p.relative_to(ROOT)}')
    print(f'[checkpoint] 产物 {len(artifacts)} 项, 结果 {len(results)} 项, 下一阶段: {args.next_start}')


# ════════════════════════════════════════════════════════════════
#  list: 列所有存档
# ════════════════════════════════════════════════════════════════
def cmd_list(args):
    data = _load(args.task)
    _, from_meta = _archive_path(args.task)
    if not from_meta:
        print(f'[checkpoint] WARN: 未找到 task_meta.json，使用当日目录，请确认 task={args.task} 启动日')
    if not data:
        print(f'[checkpoint] 任务 {args.task} 无存档点')
        return
    print(f'[checkpoint] 任务 {args.task} 共 {len(data)} 个存档点:')
    for i, r in enumerate(data):
        print(f'  #{r["stage_idx"]} [{r["time"]}] {r["stage"]}')
        if r.get('results'):
            res = ', '.join(f'{k}={v}' for k, v in r['results'].items())
            print(f'       结果: {res}')
        print(f'       下一阶段: {r.get("next_start", "(未设)")}')

    # 链一致性检查：每个 next_start 应对应下一个存档的 stage（slug/stage 拼错防护）
    print('[checkpoint] 链一致性检查:')
    broken = False
    for i in range(len(data) - 1):
        cur_next = data[i].get('next_start', '')
        nxt_stage = data[i + 1].get('stage', '')
        # 宽松匹配：next_start 是 nxt_stage 的前缀或子串即可（允许 next_start=train-b 对应 stage=train-b-mdfen）
        if cur_next and cur_next not in nxt_stage and nxt_stage not in cur_next:
            print(f'  [WARN] #{data[i]["stage_idx"]} next_start="{cur_next}" 与下一阶段 stage="{nxt_stage}" 不匹配，可能拼错续错')
            broken = True
    if not broken:
        print('  [OK] 存档链连续，next_start 与下一 stage 一致')


# ════════════════════════════════════════════════════════════════
#  show-latest: 读最近存档（启动恢复用）
# ════════════════════════════════════════════════════════════════
def cmd_show_latest(args):
    data = _load(args.task)
    _, from_meta = _archive_path(args.task)
    if not from_meta:
        print(f'[checkpoint] WARN: 未找到 task_meta.json，使用当日目录，请确认 task={args.task} 启动日')
    if not data:
        print(f'[checkpoint] 任务 {args.task} 无存档点 —— 从头开始')
        return
    latest = data[-1]
    print(f'[checkpoint] 最近存档点 #{latest["stage_idx"]}: {latest["stage"]}')
    print(f'[checkpoint] 完成时间: {latest["time"]}')
    if latest.get('artifacts'):
        print('[checkpoint] 产物位置:')
        for k, v in latest['artifacts'].items():
            print(f'  {k}: {v}')
    if latest.get('results'):
        print('[checkpoint] 数据结果:')
        for k, v in latest['results'].items():
            print(f'  {k}: {v}')
    print(f'[checkpoint] >> 续跑起点: {latest.get("next_start", "(未设)")}')
    # 机器可读行：供 LLM 启动时 parse
    print(f'[checkpoint] RESUME_FROM={latest.get("next_start", "")}')


# ════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_s = sub.add_parser('save', help='打一个阶段存档点')
    p_s.add_argument('--task', required=True, help='任务标识（与 .tmp/<日期>/<task>/ 一致）')
    p_s.add_argument('--stage', required=True, help='阶段名，如「训练A」')
    p_s.add_argument('--artifact', action='append', default=[], help='产物位置指针，格式 key=path，可多次')
    p_s.add_argument('--result', action='append', default=[], help='数据结果，格式 key=value，可多次')
    p_s.add_argument('--next_start', required=True, help='下一阶段起点（崩溃恢复从这续）')
    p_s.add_argument('--note', help='可选备注')
    p_s.set_defaults(func=cmd_save)

    p_l = sub.add_parser('list', help='列任务所有存档点')
    p_l.add_argument('--task', required=True)
    p_l.set_defaults(func=cmd_list)

    p_sl = sub.add_parser('show-latest', help='读最近存档（启动恢复用）')
    p_sl.add_argument('--task', required=True)
    p_sl.set_defaults(func=cmd_show_latest)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
