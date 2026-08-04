#!/bin/bash
# scripts/pipeline.sh — 泛化: 跑单个训练任务, retry + 终点检查 (H1)
#
# 用法 (env var):
#   RUN_CMD          必填, 训练命令 (e.g. "python train.py --cfg x.yaml")
#   DONE_MARKER      必填, 成功标志文件 (e.g. "output/x/best.pth")
#   LOG_PATH         可选, log 输出路径 (默认 /tmp/<TASK>_pipeline.log)
#   RETRY_MAX        可选, 重试次数 (默认 3)
#   NAN_PATTERN      可选, NaN 失败正则 (默认 "loss[=: ]+nan\b")
#   OOM_PATTERN      可选, OOM 失败正则 (默认 "out of memory|cuda error")
#
# 退出码:
#   0   成功 (DONE_MARKER 存在 + 无 NaN/OOM)
#   1   全部 retry 都失败
#   124 单次超时 (kill -TERM, 下次 retry)
#
# 配合 workflow: 每次 retry 由 workflow 调度, 本脚本只做单次.

set -uo pipefail

: "${RUN_CMD:?RUN_CMD required}"
: "${DONE_MARKER:?DONE_MARKER required}"
: "${LOG_PATH:=/tmp/$(basename "$0")_$$.log}"
: "${RETRY_MAX:=3}"
: "${NAN_PATTERN:=loss[=: ]+nan\\b}"
: "${OOM_PATTERN:=out of memory|cuda error}"

mkdir -p "$(dirname "$LOG_PATH")"

for try in $(seq 1 "$RETRY_MAX"); do
    echo "[pipeline] try $try/$RETRY_MAX  cmd=$RUN_CMD"
    if bash -c "$RUN_CMD" > "$LOG_PATH" 2>&1; then
        # 检查 NaN/OOM (即使 exit 0 也可能权重烂)
        if grep -aiE "$NAN_PATTERN" "$LOG_PATH" | head -5 | grep -qiE 'nan|inf'; then
            echo "[pipeline] try $try: NaN detected in log"
            [ "$try" -lt "$RETRY_MAX" ] && { sleep 5; continue; }
            echo "[pipeline] NaN persistent, FAIL"; exit 1
        fi
        if grep -aiE "$OOM_PATTERN" "$LOG_PATH" | head -5 | grep -qiE 'oom|out of memory'; then
            echo "[pipeline] try $try: OOM detected"
            [ "$try" -lt "$RETRY_MAX" ] && { sleep 10; continue; }
            echo "[pipeline] OOM persistent, FAIL"; exit 1
        fi
        # 检查 DONE_MARKER
        if [ -f "$DONE_MARKER" ]; then
            echo "[pipeline] DONE marker found: $DONE_MARKER"
            exit 0
        else
            echo "[pipeline] try $try: exit 0 但 DONE_MARKER ($DONE_MARKER) 不存在"
            [ "$try" -lt "$RETRY_MAX" ] && { sleep 5; continue; }
            echo "[pipeline] no marker, FAIL"; exit 1
        fi
    else
        RC=$?
        echo "[pipeline] try $try: exit $RC"
        [ "$try" -lt "$RETRY_MAX" ] && { sleep 5; continue; }
        exit "$RC"
    fi
done