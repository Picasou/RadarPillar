#!/bin/bash
# train_b8-bs8.sh — 自动生成 by model-train skill
# 生成时间: 2026-08-04 11:49:32
# task=train-b8-bs8
#
# 训练任务: b8  (共 1 个模型)
#   bs=8  epochs=80  gpu=0
#
# 嵌入保护机制 (skill 默认开启):
#   ✓ tmux_spawn     parent=/init (跨 Claude session 存活)
#   ✓ watchdog       driver 死了自动 tmux 重启
#   ✓ brief          cron 10min 进度简报
#   ✓ done_notifier  完成事件信号
#
# 启动方式:
#   # 1. tmux 内 (skill 已自动 tmux_spawn)
#   tmux attach -t rpillar_train-b8-bs8
#
#   # 2. 杀
#   tmux kill-session -t rpillar_train-b8-bs8
#
# 进度:
#   tail -f /tmp/train-b8-bs8.brief.out
#
# 手动重启 (driver 死了, watchdog 没救回):
#   bash /home/admin/projects/RadarPillar/.claude/skills/model-train/helpers/watchdog.sh train-b8-bs8

set -uo pipefail

TASK="train-b8-bs8"
TAGS=("b8")
BS=8
EPOCHS=80
GPU=0
SESSION="rpillar_${TASK}"
LOG="/tmp/${TASK}.log"
BRIEF_OUT="/tmp/${TASK}.brief.out"

SKILL_DIR="/home/admin/projects/RadarPillar/.claude/skills/model-train"
PIPELINE="${SKILL_DIR}/scripts/pipeline.sh"

cd "$(dirname "$0")/../.."   # 工程根
export PYTHONNOUSERSITE=1

# conda env 自适应
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
find_conda_env() {
    local try_envs=("${DESIRED_ENV:-base}" "base")
    local installed; installed="$(conda env list 2>/dev/null | awk 'NF && $1 != "#" {print $1}')"
    for env in "${try_envs[@]}"; do
        if echo "$installed" | grep -qx "$env"; then echo "$env"; return 0; fi
    done; return 1
}
TARGET_ENV="$(find_conda_env)" || { echo "[ERROR] 无可用 conda env"; exit 1; }
conda activate "$TARGET_ENV"

# driver PID 留档, watchdog 自检
DRIVER_PID=$$
echo "$DRIVER_PID" > "/tmp/${TASK}.driver.pid"

# 完成信号文件 (workflow 跑完写), done_notifier 监听
DONE_FILE="/tmp/${TASK}.done"
TS_FILE="/tmp/${TASK}.done.ts"
rm -f "$DONE_FILE" 2>/dev/null

echo "[workflow] start $(date)  TASK=${TASK}  models=${#TAGS[@]}  bs=${BS}"

# === 主循环: 顺序训练 + 健康检查 + bs-aware oracle ===
for ((i=0; i<${#TAGS[@]}; i++)); do
    tag="${TAGS[$i]}"
    if [ $((i+1)) -lt ${#TAGS[@]} ]; then nxt="${TAGS[$((i+1))]}"; else nxt="done"; fi
    echo "==========================================="
    echo "[workflow] === ${tag} (next=${nxt})  $(date)"

    # === bs-aware oracle ===
    OR=$(ls -td output/train_log/vod/*_rpillar_${tag}_${tag} 2>/dev/null | head -1) || true
    SKIP=0
    if [ -n "$OR" ] && [ -f "$OR/best.pth" ] && [ -f "$OR/model_store.yaml" ]; then
        PRIOR_LOG=$(ls -t "$OR"/train_*.log "$OR"/logs/train_*.log 2>/dev/null | head -1) || true
        PRIOR_BS=$(grep -m1 -E "INFO  batch_size" "$PRIOR_LOG" 2>/dev/null | awk '{print $NF}') || true
        if [ "$PRIOR_BS" = "${BS}" ]; then
            echo "[workflow] ${tag} oracle 命中 (prior_bs=${BS}), 跳过"
            SKIP=1
        else
            echo "[workflow] ${tag} prior_bs='${PRIOR_BS}' ≠ ${BS}, 重训"
        fi
    fi

    if [ $SKIP -eq 0 ]; then
        set +e
        MODEL=rpillar_${tag} CFG_FILE=experiments/YAML/${tag}.yaml EXTRA_TAG=${tag} \
        EPOCHS=${EPOCHS} BATCH_SIZE=${BS} GPU=${GPU} \
        LAST_N_EVAL=10 RUN_VIZ=false RUN_PICKBEST=true RUN_RESBAG=true \
            bash "$PIPELINE"
        RC=$?
        set -e 2>/dev/null || true
        if [ $RC -ne 0 ]; then
            echo "[workflow] ${tag} 失败 (rc=${RC}), continue"
        else
            echo "[workflow] ${tag} 成功"
        fi
    fi

    # 每模型完成: 健康自检
    if [ ! -d "/proc/$DRIVER_PID" ]; then
        echo "[workflow] ⚠️  driver PID=$DRIVER_PID 死了, 等 watchdog 自愈"
    fi
done

touch "$DONE_FILE"
echo "[workflow] ALL DONE $(date)  TASK=${TASK}  共 ${#TAGS[@]} 模型"
