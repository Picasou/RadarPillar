#!/bin/bash
# workflow_n1.sh — 自动生成 by model-train skill
# 生成时间: 2026-08-04 15:31:42  task=train-n1
#
# 任务数: 1  retry: 3  gpu_limit: 7000MiB
#
# 嵌入保护机制:
#   ✓ tmux_spawn     parent=/init (跨 Claude session)
#   ✓ watchdog       cron 10min 自检 + 自动重启 (含 H2 cron 守护)
#   ✓ brief          cron 10min 进度 (含 H3 GPU 显存告警)
#   ✓ done_notifier  完成校验 + 触发 post-task hook (H4 终点校验)
#   ✓ pipeline.sh    内置 retry 3 次 (H1)
#
# 监控:
#   tmux attach -t rpillar_train-n1
#   tail -f /tmp/train-n1.brief.out
# 杀: tmux kill-session -t rpillar_train-n1
# 重启: bash /home/admin/projects/RadarPillar/.claude/skills/model-train/helpers/watchdog.sh train-n1

set -uo pipefail

TASK="train-n1"
MAX_RETRY=3
GPU_LIMIT_MIB=7000
SESSION="rpillar_${TASK}"
LOG="/tmp/${TASK}.log"
BRIEF_OUT="/tmp/${TASK}.brief.out"

SKILL_DIR="/home/admin/projects/RadarPillar/.claude/skills/model-train"
PIPELINE="${SKILL_DIR}/scripts/pipeline.sh"

cd "$(dirname "$0")/../.." 2>/dev/null || cd "$HOME"

export PYTHONNOUSERSITE=1

# === 任务定义 (skill 生成时填入) ===
TASKS_SPEC=(
    "n1|bash /home/admin/projects/RadarPillar/experiments/SH/train_n1_full.sh|/home/admin/projects/RadarPillar/output/n1.done"

)

# driver PID 留档
DRIVER_PID=$$
echo "$DRIVER_PID" > "/tmp/${TASK}.driver.pid"

# 完成信号文件
DONE_FILE="/tmp/${TASK}.done"
rm -f "$DONE_FILE"

# === 主循环 ===
echo "[workflow] start $(date)  TASK=${TASK}  tasks=${#TASKS_SPEC[@]}  retry=${MAX_RETRY}"

declare -A TASK_RESULTS
PASS_COUNT=0
FAIL_COUNT=0
FAILED_TASKS=()

for spec in "${TASKS_SPEC[@]}"; do
    IFS='|' read -r tag cmd marker <<< "$spec"
    echo "==========================================="
    echo "[workflow] === ${tag}  $(date)"

    # === 终点 oracle: marker 存在则跳过 ===
    if [ -f "$marker" ]; then
        echo "[workflow] ${tag} marker 命中 ($marker), 跳过"
        TASK_RESULTS[$tag]="SKIP"
        continue
    fi

    # === pipeline.sh retry (H1) ===
    RUN_CMD="$cmd" \
    DONE_MARKER="$marker" \
    LOG_PATH="/tmp/${TASK}_${tag}.log" \
    RETRY_MAX="$MAX_RETRY" \
        bash "$PIPELINE"
    RC=$?

    if [ $RC -eq 0 ] && [ -f "$marker" ]; then
        echo "[workflow] ${tag} ✓"
        TASK_RESULTS[$tag]="OK"
        PASS_COUNT=$((PASS_COUNT+1))
    else
        echo "[workflow] ${tag} ✗ (rc=$RC marker=$marker)"
        TASK_RESULTS[$tag]="FAIL_rc${RC}"
        FAIL_COUNT=$((FAIL_COUNT+1))
        FAILED_TASKS+=("$tag")
    fi
done

# === 终点写 done 文件 (H4 由 done_notifier 校验全部 marker) ===
touch "$DONE_FILE"
echo "[workflow] DONE $(date)  TASK=${TASK}  pass=${PASS_COUNT}  fail=${FAIL_COUNT}"
[ ${#FAILED_TASKS[@]} -gt 0 ] && echo "[workflow] failed: ${FAILED_TASKS[*]}"
