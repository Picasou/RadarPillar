#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长周期任务 .tmp/ 进度文件初始化（确定性，替代大模型手写骨架）。

目录约定：.tmp/<YYYY-MM-DD>/<slug>/<slug>.md
日期为任务启动日（启动即定，跨天不挪窝），同任务的所有临时文件都放该子目录。
"""
import argparse
import json
import sys
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
    ap.add_argument('--force', action='store_true',
                    help='P1-10 修复: 同 slug 已存在 task_meta.json 时强制重入(默认拒绝)')
    args = ap.parse_args()

    start_dt = datetime.now(CST)
    start_date = start_dt.strftime('%Y-%m-%d')
    start_str = start_dt.strftime('%Y-%m-%d %H:%M CST')

    task_dir = TMP_DIR / start_date / args.slug

    # P1-10 修复: --refuse-reinit（同 slug 已存在则 fail）
    meta_path = task_dir / 'task_meta.json'
    if meta_path.exists() and not args.force:
        sys.exit(
            f'[init] [FAIL] slug={args.slug} 已存在 task_meta.json (在 {task_dir})\n'
            f'[init] 同 slug 续接请用 `checkpoint.py show-latest --task {args.slug}` 读存档\n'
            f'[init] 如确认要新开任务,加 --force 强制重入(覆盖旧 meta + 索引)'
        )

    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / f'{args.slug}.md'

    meta = {
        'slug': args.slug,
        'name': args.name,
        'start_date': start_date,
        'start_time': start_str,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    # P-partial 修复: init 时扫所有 BLOCKED.json,若有自愈标记则打印警报
    blocked_list = list(TMP_DIR.rglob('BLOCKED.json'))
    if blocked_list:
        print(f'[init] [ALERT] 发现 {len(blocked_list)} 个 BLOCKED.json(NaN/OOM 自愈标记):')
        for bp in blocked_list:
            try:
                data = json.loads(bp.read_text(encoding='utf-8'))
                print(f'  - {bp.relative_to(ROOT)}: model={data.get("model")}, reason={data.get("reason")}, ep={data.get("last_epoch")}, ckpt={data.get("last_ckpt")}')
            except Exception as e:
                print(f'  - {bp.relative_to(ROOT)}: parse error {e}')

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
