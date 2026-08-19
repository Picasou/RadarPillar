#!/bin/bash
# scripts/generate_workflow.sh — 你只说训练哪些模型(tag), skill 全自动编排
#
# 输入: tag 列表 (逗号分隔或文件, 每行一个 tag)
# skill 对每个 tag 自动:
#   1. 找 experiments/SH/train_rpillar_<tag>.sh + experiments/YAML/<tag>.yaml
#   2. 从 templates/full_chain.template.sh 生成 train_<tag>_full.sh (train+eval+pickbest+resbag)
#   3. 串行调度: marker 跳过 + retry + 结论文档
#
# 例:
#   bash scripts/generate_workflow.sh --tasks "n2,n3" --max-retry 3
#   bash scripts/generate_workflow.sh --tasks-file tags.txt
#
# tag→路径约定可覆盖: --train-dir / --yaml-dir (默认 experiments/SH / experiments/YAML)
#   训练脚本前缀可覆盖: --sh-prefix (默认 train_rpillar_, 如 MSR 用 train_)

set -uo pipefail

TASKS_STR=""
TASKS_FILE=""
MAX_RETRY=3
DATASET="vod"
TRAIN_DIR="experiments/SH"
YAML_DIR="experiments/YAML"
SH_PREFIX="train_rpillar_"
EPOCHS=80
BS=8
TASK_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)        TASKS_STR="$2"; shift 2 ;;
        --tasks-file)   TASKS_FILE="$2"; shift 2 ;;
        --max-retry)    MAX_RETRY="$2"; shift 2 ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --train-dir)    TRAIN_DIR="$2"; shift 2 ;;
        --yaml-dir)     YAML_DIR="$2"; shift 2 ;;
        --sh-prefix)    SH_PREFIX="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --bs)           BS="$2"; shift 2 ;;
        --task)         TASK_OVERRIDE="$2"; shift 2 ;;
        -h|--help)      sed -n '3,18p' "$0"; exit 0 ;;
        *) echo "[generate_workflow] ERROR: 未知参数 $1" >&2; exit 1 ;;
    esac
done

# 加载 tag 列表
TAGS=()
if [ -n "$TASKS_FILE" ]; then
    [ -f "$TASKS_FILE" ] || { echo "[generate_workflow] ERROR: 文件不存在 $TASKS_FILE" >&2; exit 1; }
    while IFS= read -r t; do
        t="${t%%#*}"; t="$(echo "$t" | xargs)"
        [ -z "$t" ] && continue
        TAGS+=("$t")
    done < "$TASKS_FILE"
elif [ -n "$TASKS_STR" ]; then
    IFS=',' read -ra TAGS <<< "$TASKS_STR"
else
    echo "[generate_workflow] ERROR: --tasks 或 --tasks-file 必填" >&2; exit 1
fi

[ ${#TAGS[@]} -eq 0 ] && { echo "[generate_workflow] ERROR: tasks 为空" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"
TEMPLATE="${SKILL_DIR}/templates/full_chain.template.sh"
SH_DIR="${ROOT}/${TRAIN_DIR#../}"
YAML_ROOT="${ROOT}/${YAML_DIR#../}"
if [ -n "$TASK_OVERRIDE" ]; then
    TASK="$TASK_OVERRIDE"
else
    TAGS_CONCAT="$(printf '%s' "${TAGS[@]}")"
    TASK="train-${TAGS_CONCAT}"
fi
# 生成的中间脚本一律进 .tmp/, 不污染正式目录 (experiments/ tools/ 项目根)
RUN_DIR="$ROOT/.tmp/model-train/${TASK#train-}"
mkdir -p "$RUN_DIR"
WORKFLOW_SH="$RUN_DIR/workflow_${TASK#train-}.sh"
LOG="/tmp/${TASK}.log"
BRIEF_OUT="/tmp/${TASK}.brief.out"

echo "[generate_workflow] TASK=$TASK  tags=(${TAGS[*]})  retry=$MAX_RETRY  epochs=$EPOCHS  bs=$BS"

# === 对每个 tag: 校验 .sh/.yaml 在 + 从模板生成 _full.sh (进 .tmp/) ===
FULL_SCRIPTS=()
for tag in "${TAGS[@]}"; do
    model="rpillar_${tag}"
    sh="${SH_DIR}/${SH_PREFIX}${tag}.sh"
    cfg="${YAML_ROOT}/${tag}.yaml"
    [ -f "$sh" ]  || { echo "[generate_workflow] ERROR: 训练脚本不在 $sh" >&2; exit 1; }
    [ -f "$cfg" ] || { echo "[generate_workflow] ERROR: cfg 不在 $cfg" >&2; exit 1; }

    full="${RUN_DIR}/train_${tag}_full.sh"
    echo "[generate_workflow] 生成 $full  (cfg=$cfg)"
    sed -e "s#__TAG__#$tag#g" \
        -e "s#__MODEL__#$model#g" \
        -e "s#__CFG__#$cfg#g" \
        -e "s#__FULL_FILENAME__#$(basename "$full")#g" \
        "$TEMPLATE" > "$full"
    chmod +x "$full"
    FULL_SCRIPTS+=("$full")
done

# === 写 workflow (循环内置 marker 跳过 + retry, 直接调 _full.sh) ===
cat > "$WORKFLOW_SH" <<'OUTER_EOF'
#!/bin/bash
# __WF_FILENAME__ — 自动生成 by model-train skill
# 生成时间: __GEN_TIME__  task=__TASK__
# 串行调各模型 _full.sh (train+eval+pickbest[max+median]+resbag); marker 跳过+retry; 全完出结论文档
#
# 监控:
#   tmux attach -t rpillar___TASK__
#   tail -f /tmp/__TASK__.brief.out
# 杀: tmux kill-session -t rpillar___TASK__
# 重启: bash __SKILL_DIR__/helpers/watchdog.sh __TASK__

set -uo pipefail

TASK="__TASK__"
MAX_RETRY=__MAX_RETRY__
SESSION="rpillar_${TASK}"
SKILL_DIR="__SKILL_DIR__"
ROOT="__ROOT__"
CONCLUSION="${SKILL_DIR}/scripts/make_conclusion.py"
REPORT_TPL="${SKILL_DIR}/实验报告模板.md"

export PYTHONNOUSERSITE=1

FULL_SCRIPTS=(
__TASKS_LINES__
)

DRIVER_PID=$$
echo "$DRIVER_PID" > "/tmp/${TASK}.driver.pid"
DONE_FILE="/tmp/${TASK}.done"
rm -f "$DONE_FILE"

derive_tag() { local b; b="$(basename "$1")"; b="${b#train_}"; printf '%s' "${b%_full.sh}"; }

run_model() {   # run_model <full.sh> <tag>: marker 跳过 + retry N
    local fs="$1" tag="$2" marker="${ROOT}/output/$2.done" try rc
    for try in $(seq 1 "$MAX_RETRY"); do
        [ -f "$marker" ] && { echo "[workflow] ${tag} marker 命中, 跳过"; return 0; }
        echo "[workflow] ${tag} try ${try}/${MAX_RETRY}"
        bash "$fs" && [ -f "$marker" ] && { echo "[workflow] ${tag} ✓"; return 0; }
        rc=$?; echo "[workflow] ${tag} ✗ (rc=$rc 或 marker 缺)"
        [ "$try" -lt "$MAX_RETRY" ] && sleep 5
    done
    echo "[workflow] ${tag} 最终失败"; return 1
}

PASS_COUNT=0; FAIL_COUNT=0; FAILED_TASKS=()
echo "[workflow] start $(date)  TASK=${TASK}  tasks=${#FULL_SCRIPTS[@]}  retry=${MAX_RETRY}"

for fs in "${FULL_SCRIPTS[@]}"; do
    tag="$(derive_tag "$fs")"
    echo "==========================================="
    echo "[workflow] === ${tag}  $(date)"
    if run_model "$fs" "$tag"; then PASS_COUNT=$((PASS_COUNT+1));
    else FAIL_COUNT=$((FAIL_COUNT+1)); FAILED_TASKS+=("$tag"); fi
done

# === 综合报告: make_conclusion 数据表 + 套报告模板骨架 ===
REPORT="${ROOT}/output/train_log/__DATASET__/_report_${TASK#train-}.md"
python "$CONCLUSION" --dataset "__DATASET__" --root "${ROOT}/output/train_log" \
    -o "$REPORT" 2>&1 | tail -1 || echo "[workflow] 数据表生成失败 (不阻塞)"
# 报告模板骨架附在数据表后 (判断段留给 LLM/人填)
[ -f "$REPORT_TPL" ] && printf '\n\n---\n## 报告结构参照 (模板)\n\n' >> "$REPORT" && cat "$REPORT_TPL" >> "$REPORT"

touch "$DONE_FILE"
echo "[workflow] DONE $(date)  pass=${PASS_COUNT}  fail=${FAIL_COUNT}  report=$REPORT"
[ ${#FAILED_TASKS[@]} -gt 0 ] && echo "[workflow] failed: ${FAILED_TASKS[*]}"
OUTER_EOF

# 路径转绝对喂给 workflow
TASKS_LINES=""
for fs in "${FULL_SCRIPTS[@]}"; do
    case "$fs" in /*) abs="$fs";; *) abs="$ROOT/$fs";; esac
    TASKS_LINES="${TASKS_LINES}    \"${abs}\"\n"
done
TASKS_LINES="${TASKS_LINES%$'\n'}"

sed -i \
    -e "s#__TASK__#$TASK#g" \
    -e "s#__WF_FILENAME__#workflow_${TASK#train-}.sh#g" \
    -e "s#__DATASET__#$DATASET#g" \
    -e "s#__MAX_RETRY__#$MAX_RETRY#g" \
    -e "s#__GEN_TIME__#$(date '+%F %T')#g" \
    -e "s#__SKILL_DIR__#$SKILL_DIR#g" \
    -e "s#__ROOT__#$ROOT#g" \
    "$WORKFLOW_SH"

awk -v tasks="$TASKS_LINES" '/^__TASKS_LINES__$/ {print tasks; next} {print}' \
    "$WORKFLOW_SH" > "${WORKFLOW_SH}.tmp" && mv "${WORKFLOW_SH}.tmp" "$WORKFLOW_SH"

chmod +x "$WORKFLOW_SH"

# === H5: 语法检查 (workflow + 各 _full.sh) ===
ERR=0
bash -n "$WORKFLOW_SH" 2>/dev/null || { echo "[generate_workflow] ❌ workflow 语法错"; ERR=1; }
for fs in "${FULL_SCRIPTS[@]}"; do
    bash -n "$fs" 2>/dev/null || { echo "[generate_workflow] ❌ $fs 语法错"; ERR=1; }
done
[ $ERR -eq 0 ] || exit 1

echo "[generate_workflow] ✓ 生成 ${#TAGS[@]} 个 _full.sh + workflow, 语法全过"
echo ""
echo "下一步 (skill 调用方):"
echo "  bash $SKILL_DIR/helpers/tmux_spawn.sh rpillar_$TASK \"$ROOT\" \\"
echo "      \"bash $WORKFLOW_SH 2>&1 | tee $LOG\""
echo "  ( crontab -l 2>/dev/null"
echo "    echo \"*/10 * * * * bash $SKILL_DIR/scripts/brief.sh $TASK $BRIEF_OUT >> $LOG 2>&1\""
echo "    echo \"*/10 * * * * sleep 30 && bash $SKILL_DIR/helpers/watchdog.sh $TASK $WORKFLOW_SH\""
echo "  ) | crontab -"
