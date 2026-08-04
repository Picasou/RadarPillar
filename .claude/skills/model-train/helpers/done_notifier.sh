#!/bin/bash
# helpers/done_notifier.sh — 泛化: 完成校验 + 触发 post-task hook (H4)
#
# 用法 (常驻 daemon, 后台跑):
#   bash helpers/done_notifier.sh <DONE_FILE> <TS_FILE> <STABLE_MIN> <MARKERS_FILE>
#   # MARKERS_FILE: 每行一个 marker 路径, done_notifier 校验全齐才标 complete
#   # POST_HOOK env: 完成后跑的脚本 (e.g. 写报告 / 发通知 / 跑下一阶段)
#
# 退出码:
#   0  complete (全 marker 齐, post-hook 跑过)
#   1  partial  (缺 marker)
#   2  错误
#
# 与 watchdog 的区别:
#   done_notifier = 终点校验 (事件触发, 一次性)
#   watchdog      = 健康守护 (持续自检, 自动重启)
#   两者互补, 不重复

set -uo pipefail

DONE_FILE="${1:?done file path required}"
TS_FILE="${2:?ts file required}"
STABLE_MIN="${3:-5}"
MARKERS_FILE="${4:-}"

POST_HOOK="${POST_HOOK:-}"
mkdir -p "$(dirname "$TS_FILE")"

if [ -f "$TS_FILE" ] && grep -q "^complete" "$TS_FILE" 2>/dev/null; then
    echo "[done_notifier] $TS_FILE 已标 complete, exit 0"
    exit 0
fi

PREV_MTIME=""
STABLE_COUNT=0
STABLE_MAX=$(( STABLE_MIN * 2 ))   # 30s/poll

while true; do
    [ -f "$DONE_FILE" ] || { sleep 30; continue; }

    CUR_MTIME=$(stat -c '%Y' "$DONE_FILE" 2>/dev/null || echo "")
    [ -n "$CUR_MTIME" ] || { sleep 30; continue; }

    if [ "$CUR_MTIME" = "$PREV_MTIME" ]; then
        STABLE_COUNT=$((STABLE_COUNT+1))
        if [ $STABLE_COUNT -ge $STABLE_MAX ]; then
            # === H4: 校验所有 marker (前提: MARKERS_FILE 提供) ===
            if [ -n "$MARKERS_FILE" ] && [ -f "$MARKERS_FILE" ]; then
                MISSING=""
                while IFS= read -r marker; do
                    [ -z "$marker" ] && continue
                    [ -f "$marker" ] || MISSING="$MISSING $marker"
                done < "$MARKERS_FILE"
                if [ -n "$MISSING" ]; then
                    echo "[done_notifier] partial: 缺 marker:$MISSING"
                    echo "partial" > "$TS_FILE"
                    exit 1
                fi
            fi

            echo "[done_notifier] complete: 全 marker 齐, mtime=$CUR_MTIME"
            echo "complete $CUR_MTIME" > "$TS_FILE"

            # === 触发 post-task hook ===
            if [ -n "$POST_HOOK" ]; then
                echo "[done_notifier] post-hook: $POST_HOOK"
                if ! bash -c "$POST_HOOK"; then
                    echo "[done_notifier] post-hook 失败 (rc=$?), 仍标 complete"
                fi
            fi
            exit 0
        fi
    else
        STABLE_COUNT=0
        PREV_MTIME="$CUR_MTIME"
    fi
    sleep 30
done