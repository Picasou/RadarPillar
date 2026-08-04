#!/bin/bash
# scripts/brief.sh — 泛化: 进度简报 + H3 GPU 显存告警 + driver 健康
#
# 用法 (cron 每 10min):
#   bash scripts/brief.sh <TASK_NAME> [BRIEF_OUT] [LOG_PATTERN] [EPOCH_PATTERN]
#
# TASK_NAME     e.g. train-b1b8n1n6n8
# BRIEF_OUT     默认 /tmp/<TASK>.brief.out
# LOG_PATTERN   默认 *.log (扫 output/ 下最新)
# EPOCH_PATTERN 默认 "epochs?:\s*\d+%\|.*?(\d+)/(\d+)\s*\[" (tqdm 风格)

set -eo pipefail

TASK="${1:?task name required}"
BRIEF_OUT="${2:-/tmp/${TASK}.brief.out}"
LOG_PATTERN="${3:-*.log}"
EPOCH_PATTERN="${4:-epochs?:[[:space:]]*\\d+%\\|.*?([0-9]+)/([0-9]+)[[:space:]]*\\[}"

cd "$(cd "$(dirname "$0")/../../../.." && pwd)"

SESSION="rpillar_${TASK}"

# === driver + train 进程健康 ===
DRIVER_PID=""
TRAIN_PID=""
if tmux has-session -t "$SESSION" 2>/dev/null; then
    PANE_PID=$(tmux list-panes -t "$SESSION" -F "#{pane_pid}" 2>/dev/null | head -1)
    [ -n "$PANE_PID" ] && [ -d "/proc/$PANE_PID" ] && DRIVER_PID="$PANE_PID"
fi
TRAIN_PID=$(ps -ef | awk '/train|python/ && !/awk/ && !/brief/ {print $2; exit}')

NOW=$(date '+%Y-%m-%d %H:%M CST')
{
    if [ -n "$TRAIN_PID" ]; then
        :  # 训练中
    elif [ -n "$DRIVER_PID" ]; then
        echo "[$NOW] brief: DRIVER ALIVE 但无 train.py (eval/收尾间隔期)"
    else
        echo "[$NOW] brief: ❌ DRIVER DEAD, 查 /tmp/${TASK}.log"
    fi

    # === H3: GPU 显存告警 ===
    if command -v nvidia-smi >/dev/null 2>&1; then
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | awk '{print $1}')
        if [ -n "$USED" ] && [ "${USED%.*}" -gt 7000 ]; then
            echo "[$NOW] brief: ⚠️  GPU 显存 ${USED}MiB >7G, OOM 风险"
        fi
    fi
} >> "$BRIEF_OUT"

# === 找最新 log (兜底: 扫 cwd 下所有 *.log) ===
LATEST_LOG=""
LATEST_MTIME=0
while IFS= read -r log; do
    mt=$(stat -c %Y "$log" 2>/dev/null || echo 0)
    [ "$mt" -gt "$LATEST_MTIME" ] && { LATEST_MTIME=$mt; LATEST_LOG="$log"; }
done < <(find . -name "$LOG_PATTERN" -type f 2>/dev/null | head -50)

[ -n "$LATEST_LOG" ] || { echo "[$NOW] brief: 无活跃 log" >> "$BRIEF_OUT"; exit 0; }

# 30min 无进展告警
CKPT_AGE=$(( $(date +%s) - LATEST_MTIME ))
[ "$CKPT_AGE" -gt 1800 ] && echo "[$NOW] brief: 已 $((CKPT_AGE/60))min 无进展, 可能卡死" >> "$BRIEF_OUT"

echo "[brief] log=$LATEST_LOG" >> "$BRIEF_OUT"
LOG="$LATEST_LOG" EPOCH_PATTERN="$EPOCH_PATTERN" python3 <<'PY' >> "$BRIEF_OUT"
import os, re
from datetime import datetime, timezone, timedelta
CST = timezone(timedelta(hours=8))
text = open(os.environ['LOG'], errors='ignore').read()
ep_re = re.compile(os.environ['EPOCH_PATTERN'])
loss_re = re.compile(r'loss[=: ]+([0-9.eE+-]+)', re.I)
lr_re   = re.compile(r'lr[=: ]+([0-9.eE+-]+)', re.I)
nan_re  = re.compile(r'loss[=: ]+nan\b', re.I)
oom_re  = re.compile(r'out of memory|cuda error', re.I)
cur, total = None, None
last_loss = last_lr = None
for line in text.splitlines():
    m = ep_re.search(line)
    if m: cur, total = int(m.group(1)), int(m.group(2))
    m = loss_re.search(line)
    if m: last_loss = m.group(1)
    m = lr_re.search(line)
    if m: last_lr = m.group(1)
prog = f'{cur}/{total}' if cur else 'N/A'
now = datetime.now(CST).strftime('%Y-%m-%d %H:%M')
print(f'[{now} CST] ep{prog} | loss={last_loss or "N/A"} lr={last_lr or "N/A"} | nan={int(bool(nan_re.search(text)))} oom={int(bool(oom_re.search(text)))}')
PY