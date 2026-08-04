#!/bin/bash
# helpers/watchdog.sh — 泛化: 健康守护 (含 H2 cron 自我守护)
#
# 用法 (cron 每 10min):
#   bash helpers/watchdog.sh <TASK_NAME> [WORKFLOW_SCRIPT]
#
# TASK = train-<something>  (e.g. train-b1b8n1n6n8)
# WORKFLOW_SCRIPT 默认 = workflow_<task#train->.sh (相对 cwd)
#
# 检查项:
#   1. tmux session 是否在
#   2. tmux pane bash 是否在
#   3. cron 守护进程是否在 (H2)
#
# 任一失败 → tmux_spawn.sh 重启

set -uo pipefail

TASK="${1:?task name required}"
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="/tmp/${TASK}.watchdog.log"
WORKFLOW="${2:-workflow_${TASK#train-}.sh}"

SESSION="rpillar_${TASK}"

# === H2: cron 自我守护 ===
CRON_ALIVE=false
if pgrep -x cron >/dev/null 2>&1 || pgrep -x crond >/dev/null 2>&1; then
    CRON_ALIVE=true
else
    echo "[$(date '+%F %T')] watchdog: ❌ cron 死了, 尝试重启" >> "$LOG"
    if command -v systemctl >/dev/null; then
        systemctl restart cron 2>/dev/null || systemctl restart crond 2>/dev/null
    elif command -v service >/dev/null; then
        service cron restart 2>/dev/null || service crond restart 2>/dev/null
    fi
    sleep 2
    if pgrep -x cron >/dev/null 2>&1 || pgrep -x crond >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] watchdog: ✓ cron 已重启" >> "$LOG"
    else
        echo "[$(date '+%F %T')] watchdog: ⚠️  cron 重启失败, brief/watchdog 可能不跑" >> "$LOG"
    fi
fi

# === driver 健康检查 ===
DRIVER_ALIVE=false
if tmux has-session -t "$SESSION" 2>/dev/null; then
    PANE_PID=$(tmux list-panes -t "$SESSION" -F "#{pane_pid}" 2>/dev/null | head -1)
    if [ -n "$PANE_PID" ] && [ -d "/proc/$PANE_PID" ]; then
        DRIVER_ALIVE=true
    fi
fi

if $DRIVER_ALIVE && $CRON_ALIVE; then
    echo "[$(date '+%F %T')] watchdog: ✓ 全活 (driver+cron), 不动"
    exit 0
fi

# === 需要重启 ===
{
    echo "[$(date '+%F %T')] watchdog: ❌ 异常 driver=$DRIVER_ALIVE cron=$CRON_ALIVE, 重启 workflow"
    echo "[$(date '+%F %T')] watchdog: workflow=$WORKFLOW"
} >> "$LOG"

if [ ! -f "$WORKFLOW" ]; then
    echo "[$(date '+%F %T')] watchdog: ERROR workflow=$WORKFLOW 不存在" >> "$LOG"
    exit 2
fi

bash "$SKILL_DIR/helpers/tmux_spawn.sh" "$SESSION" "$(pwd)" \
    "bash $WORKFLOW 2>&1 | tee /tmp/${TASK}.log" >> "$LOG" 2>&1

echo "[$(date '+%F %T')] watchdog: 重启已发, 下 tick 验证"