#!/bin/bash
# scripts/generate_workflow.sh — 泛化: 用户任务列表 → workflow 脚本
#
# 用法:
#   # 简单格式 (逗号分隔, 每项 "tag|cmd|marker")
#   bash scripts/generate_workflow.sh \
#       --tasks "b1|python train.py --cfg b1.yaml|output/b1/done,b8|python train.py --cfg b8.yaml|output/b8/done" \
#       --max-retry 3
#
#   # 文件格式 (每行: tag<TAB>cmd<TAB>marker)
#   bash scripts/generate_workflow.sh --tasks-file tasks.tsv
#
# 生成的 workflow 嵌入 (skill 默认):
#   - tmux_spawn: parent=/init, 跨 Claude session 存活
#   - watchdog:  driver 死了自动 tmux 重启 (cron 10min)
#   - brief:     cron 10min 进度简报 (含 H3 GPU 显存告警)
#   - done_notifier: 监听完成 + 触发 post-task hook (H4 终点校验)
#   - pipeline.sh retry: 每任务 N 次 (H1)

set -uo pipefail

TASKS_STR=""
TASKS_FILE=""
MAX_RETRY=3
GPU_LIMIT=7000
TASK_OVERRIDE=""
GPU_FLAG=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)        TASKS_STR="$2"; shift 2 ;;
        --tasks-file)   TASKS_FILE="$2"; shift 2 ;;
        --max-retry)    MAX_RETRY="$2"; shift 2 ;;
        --gpu-limit)    GPU_LIMIT="$2"; shift 2 ;;
        --no-gpu)       GPU_FLAG=false; shift ;;
        --task)         TASK_OVERRIDE="$2"; shift 2 ;;
        -h|--help)      sed -n '3,18p' "$0"; exit 0 ;;
        *) echo "[generate_workflow] ERROR: 未知参数 $1" >&2; exit 1 ;;
    esac
done

# 加载任务列表
TASKS=()
if [ -n "$TASKS_FILE" ]; then
    [ -f "$TASKS_FILE" ] || { echo "[generate_workflow] ERROR: 文件不存在 $TASKS_FILE" >&2; exit 1; }
    while IFS=$'\t' read -r tag cmd marker; do
        [ -z "$tag" ] && continue
        TASKS+=("$tag|$cmd|$marker")
    done < "$TASKS_FILE"
elif [ -n "$TASKS_STR" ]; then
    IFS=',' read -ra TASKS <<< "$TASKS_STR"
else
    echo "[generate_workflow] ERROR: --tasks 或 --tasks-file 必填" >&2; exit 1
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "[generate_workflow] ERROR: tasks 为空" >&2; exit 1
fi

# TASK 命名
if [ -n "$TASK_OVERRIDE" ]; then
    TASK="$TASK_OVERRIDE"
else
    TAGS_CONCAT=$(printf '%s' "${TASKS[@]%%|*}" | tr -d ' \n' | sed 's/|//g')
    TASK="train-${TAGS_CONCAT}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
WORKFLOW_SH="$ROOT/workflow_${TASK#train-}.sh"
LOG="/tmp/${TASK}.log"
BRIEF_OUT="/tmp/${TASK}.brief.out"

echo "[generate_workflow] TASK=$TASK  tasks=${#TASKS[@]}  retry=$MAX_RETRY"
echo "[generate_workflow] → $WORKFLOW_SH"

# === 写入 workflow (heredoc 关闭, 字面写入) ===
cat > "$WORKFLOW_SH" <<'OUTER_EOF'
#!/bin/bash
# __WF_FILENAME__ — 自动生成 by model-train skill
# 生成时间: __GEN_TIME__  task=__TASK__
#
# 任务数: __N_TASKS__  retry: __MAX_RETRY__  gpu_limit: __GPU_LIMIT__MiB
#
# 嵌入保护机制:
#   ✓ tmux_spawn     parent=/init (跨 Claude session)
#   ✓ watchdog       cron 10min 自检 + 自动重启 (含 H2 cron 守护)
#   ✓ brief          cron 10min 进度 (含 H3 GPU 显存告警)
#   ✓ done_notifier  完成校验 + 触发 post-task hook (H4 终点校验)
#   ✓ pipeline.sh    内置 retry __MAX_RETRY__ 次 (H1)
#
# 监控:
#   tmux attach -t rpillar___TASK__
#   tail -f /tmp/__TASK__.brief.out
# 杀: tmux kill-session -t rpillar___TASK__
# 重启: bash __SKILL_DIR__/helpers/watchdog.sh __TASK__

set -uo pipefail

TASK="__TASK__"
MAX_RETRY=__MAX_RETRY__
GPU_LIMIT_MIB=__GPU_LIMIT__
SESSION="rpillar_${TASK}"
LOG="/tmp/${TASK}.log"
BRIEF_OUT="/tmp/${TASK}.brief.out"

SKILL_DIR="__SKILL_DIR__"
PIPELINE="${SKILL_DIR}/scripts/pipeline.sh"

cd "$(dirname "$0")/../.." 2>/dev/null || cd "$HOME"

export PYTHONNOUSERSITE=1

# === 任务定义 (skill 生成时填入) ===
TASKS_SPEC=(
__TASKS_LINES__
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
OUTER_EOF

# 占位符替换: 单值用 sed, TASKS_LINES 用 awk (含 \n 真换行)
sed -i \
    -e "s#__TASK__#$TASK#g" \
    -e "s#__WF_FILENAME__#workflow_${TASK#train-}.sh#g" \
    -e "s#__N_TASKS__#${#TASKS[@]}#g" \
    -e "s#__MAX_RETRY__#$MAX_RETRY#g" \
    -e "s#__GPU_LIMIT__#$GPU_LIMIT#g" \
    -e "s#__GEN_TIME__#$(date '+%F %T')#g" \
    -e "s#__SKILL_DIR__#$SKILL_DIR#g" \
    "$WORKFLOW_SH"

# TASKS_LINES 用 awk 替换 (sed 不接受真换行的替换值)
TASKS_LINES=""
for spec in "${TASKS[@]}"; do
    TASKS_LINES="${TASKS_LINES}    \"${spec}\"\n"
done
TASKS_LINES="${TASKS_LINES%$'\n'}"   # 去尾换行

awk -v tasks="$TASKS_LINES" '
    /^__TASKS_LINES__$/ { print tasks; next }
    { print }
' "$WORKFLOW_SH" > "${WORKFLOW_SH}.tmp" && mv "${WORKFLOW_SH}.tmp" "$WORKFLOW_SH"

chmod +x "$WORKFLOW_SH"

# === H5: workflow 语法检查 ===
if ! bash -n "$WORKFLOW_SH" 2>/dev/null; then
    echo "[generate_workflow] ❌ workflow 语法错, 查 $WORKFLOW_SH"
    exit 1
fi

echo "[generate_workflow] ✓ 写入 + 语法检查通过"
echo ""
echo "下一步 (skill 调用方):"
echo "  bash $SKILL_DIR/helpers/tmux_spawn.sh rpillar_$TASK \"$ROOT\" \\"
echo "      \"bash $WORKFLOW_SH 2>&1 | tee $LOG\""
echo "  echo '*/10 * * * * bash $SKILL_DIR/scripts/brief.sh $TASK $BRIEF_OUT >> $LOG' | crontab -"