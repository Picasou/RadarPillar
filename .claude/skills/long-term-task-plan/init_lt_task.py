#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长周期任务 .tmp/ 进度文件初始化（确定性，替代大模型手写骨架）。

目录约定：.tmp/<YYYY-MM-DD>/<slug>/<slug>.md
日期为任务启动日（启动即定，跨天不挪窝），同任务的所有临时文件都放该子目录。
"""
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = ROOT / '.tmp'

SCHEMA = """# 任务：{name}
- 任务标识：{slug}
- 启动时间：{start}
- 当前阶段：（待填）
- 已完成：
- 待办：
- 关键决策：
- 决策快照（异步否决区）：   <!-- 每个替用户拍板的默认值列这里，用户可随时否决，不阻塞执行 -->
- 新增函数清单：             <!-- 本任务期间免逐个确认，事后审计用 -->
- 阻塞 / 自愈记录：
- 下次简报时间：（待填）
- 入参快照：
{inputs}
"""


def main():
    ap = argparse.ArgumentParser(description='初始化长任务 .tmp/ 进度文件')
    ap.add_argument('--name', required=True, help='任务名')
    ap.add_argument('--slug', required=True, help='任务标识（目录/文件名用）')
    ap.add_argument('--input', action='append', default=[], help='入参快照，可多次，格式 key=value')
    args = ap.parse_args()

    start_dt = datetime.now(CST)
    start_date = start_dt.strftime('%Y-%m-%d')
    start_str = start_dt.strftime('%Y-%m-%d %H:%M CST')

    task_dir = TMP_DIR / start_date / args.slug
    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / f'{args.slug}.md'

    meta_path = task_dir / 'task_meta.json'
    meta = {
        'slug': args.slug,
        'name': args.name,
        'start_date': start_date,
        'start_time': start_str,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    inputs = '\n'.join(f'  - {i}' for i in args.input) if args.input else '  （无）'

    content = SCHEMA.format(
        name=args.name,
        slug=args.slug,
        start=start_str,
        inputs=inputs,
    )
    out.write_text(content, encoding='utf-8')
    print(f'[init] 已生成 {out.relative_to(ROOT)}')
    print(f'[init] task_dir = {task_dir.relative_to(ROOT)}')
    print(f'[init] task_meta = {meta_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
