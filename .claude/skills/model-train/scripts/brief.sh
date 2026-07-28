#!/bin/bash
# brief.sh — 训练 log 10 分钟简报（cron 调用）
#
# usage:
#   bash experiments/SH/brief.sh <LOG_PATH> [OUTPUT_ROOT]
#
# 输出两行：
#   [<time> CST] ep<cur/total> | loss=<x> lr=<y> | ETA=<> | nan=<bool> oom=<bool>
#   ETA <详细>

set -eo pipefail

LOG="${1:?log path required}"
OUTPUT_ROOT="${2:-}"

# env activation
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1
    conda activate angle 2>/dev/null || true
fi
export PYTHONNOUSERSITE=1

LOG="$LOG" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
python3 <<'PY'
import os, re, sys, glob
from datetime import datetime, timezone, timedelta
CST = timezone(timedelta(hours=8))

log = os.environ['LOG']
text = open(log, errors='ignore').read()

epoch_re = re.compile(r'epochs?:\s*\d+%\|.{0,40}?\|\s*(\d+)/(\d+)\s*\[', re.I)
loss_re  = re.compile(r'loss[=: ]+([0-9.eE+-]+)', re.I)
lr_re    = re.compile(r'lr[=: ]+([0-9.eE+-]+)', re.I)
nan_re   = re.compile(r'loss[=: ]+nan\b|got ?nan|nan ?loss', re.I)
oom_re   = re.compile(r'out of memory|cuda ?error', re.I)

cur_ep = total_ep = None
last_loss = last_lr = None
# open() text mode 已把 \r\n / \r 都翻译成 \n; splitlines() 一次足够
seen_lines = text.splitlines()
for line in seen_lines:
    m = epoch_re.search(line)
    if m:
        cur_ep, total_ep = int(m.group(1)), int(m.group(2))
    m = loss_re.search(line)
    if m:
        last_loss = m.group(1)
    m = lr_re.search(line)
    if m:
        last_lr = m.group(1)

has_nan = bool(nan_re.search(text))
has_oom = bool(oom_re.search(text))

# ETA: 用最后两个 ckpt 的 mtime 间隔推剩余时间
eta = 'N/A'
if cur_ep and total_ep and cur_ep > 0 and cur_ep < total_ep:
    output_root = os.environ.get('OUTPUT_ROOT', '')
    ckpt_dir = os.path.join(output_root, 'ckpt') if output_root else ''
    if ckpt_dir and os.path.isdir(ckpt_dir):
        cks = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pth')),
                     key=os.path.getmtime)
        if len(cks) >= 2:
            dt = os.path.getmtime(cks[-1]) - os.path.getmtime(cks[-2])
            if 0 < dt < 1800:  # <30min，>0 视为有效间隔
                remaining = (total_ep - cur_ep) * dt
                eta = f'{int(remaining//3600)}h{int(remaining%3600//60)}m ≈完成 {datetime.now(CST) + timedelta(seconds=remaining):%m-%d %H:%M}'

now = datetime.now(CST).strftime('%Y-%m-%d %H:%M')
progress = f'{cur_ep}/{total_ep}' if cur_ep is not None else 'N/A'
print(f'[{now} CST] ep{progress} | loss={last_loss or "N/A"} lr={last_lr or "N/A"} | ETA={eta} | nan={int(has_nan)} oom={int(has_oom)}')
PY
