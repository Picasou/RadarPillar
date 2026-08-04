#!/bin/bash
# helpers/tmux_spawn.sh — 在独立 tmux session 内跑脚本, parent=/init (容器级)
#
# 用法:
#   bash helpers/tmux_spawn.sh <session_name> <cwd> <script_cmd...>
#
# 例:
#   bash helpers/tmux_spawn.sh rpillar_train /home/admin/projects/RadarPillar \
#       "bash experiments/SH/train_workflow.sh 2>&1 | tee /tmp/train_workflow.log"
#
# 为何: Claude Bash tool 启动的 setsid/nohup 子进程会被 reap. tmux server
#       parent=/init, WSL 容器 init 长期存活 (e.g. done_notifier.sh 6 天未死)
#
# attach 看: tmux attach -t <session_name>
# detach: Ctrl-b d
# 杀: tmux kill-session -t <session_name>

set -uo pipefail

SESSION="${1:?session name required}"
CWD="${2:?cwd required}"
shift 2
CMD="$*"

if [ -z "$CMD" ]; then
    echo "[tmux_spawn] CMD empty, exit"; exit 1
fi

# 已存在同名 session? 不重建
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[tmux_spawn] session '$SESSION' 已存在, 不重建"
    echo "[tmux_spawn] attach: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -c "$CWD" "$CMD"
sleep 1

# 验证 session 真起来了
if tmux has-session -t "$SESSION" 2>/dev/null; then
    PANE_PID=$(tmux list-panes -t "$SESSION" -F "#{pane_pid}" | head -1)
    echo "[tmux_spawn] session '$SESSION' up, pane PID=$PANE_PID"
    echo "[tmux_spawn] attach: tmux attach -t $SESSION"
    echo "[tmux_spawn] kill:   tmux kill-session -t $SESSION"
else
    echo "[tmux_spawn] ERROR: session '$SESSION' 启动失败"
    exit 2
fi