#!/bin/bash
# train_rpillar_b3.sh — 训练入口 (auto-generated from train.sh.template)
#
# 公共模板: .claude/skills/model-train/templates/train.sh.template
# 想改: 先改模板, 再 regen 全部 train_rpillar_*.sh
#
# 用法 (前台):
#   bash experiments/SH/train_rpillar_b3.sh
# 用法 (后台):
#   nohup bash experiments/SH/train_rpillar_b3.sh > /tmp/b3.log 2>&1 &

# ============================================================
#  默认值 (env 覆盖优先)
# ============================================================
# ============================================================
#  默认值 (env 覆盖优先)
# ============================================================
: "${MODEL:=rpillar_a4_lnpost}"
: "${CFG_FILE:=experiments/YAML/b3.yaml}"
: "${EXTRA_TAG:=rp_base_0716}"
: "${BATCH_SIZE:=16}"
: "${WORKERS:=2}"
: "${EPOCHS:=80}"
: "${GPU:=0}"
: "${OUTPUT_ROOT:=output/train_log/vod/$(date +%Y%m%d%H%M)_${EXTRA_TAG}_${EXTRA_TAG}}"

# 训练期不 eval (交由 unified pipeline 末 step 跑); warmup 关
SKIP_EVAL=${SKIP_EVAL:-True}
: "${RUN_MODE:=background}"
SET_CFGS=(${SET_CFGS[@]:-"OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False"})

# ============================================================
#  env activation
# ============================================================
export PYTHONNOUSERSITE=1  # 屏蔽 user-local 坏 torch (见 memory: torch-user-local-mask)

# 自动 cd 到仓库根 (此模板在 .claude/skills/model-train/templates/, 三级之上)
cd "$(dirname "$0")/../.."

# conda 自探测 (不写死)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi

find_conda_env() {
    local try_envs=("${DESIRED_ENV:-angle}" "angle" "base")
    local installed; installed="$(conda env list 2>/dev/null | awk 'NF && $1 != "#" {print $1}')"
    for env in "${try_envs[@]}"; do
        if echo "$installed" | grep -qx "$env"; then echo "$env"; return 0; fi
    done; return 1
}
TARGET_ENV="$(find_conda_env)" || { echo "[ERROR] 无可用 conda env"; exit 1; }
conda activate "$TARGET_ENV"
echo "[train] MODEL=$MODEL conda=$TARGET_ENV"

export CUDA_VISIBLE_DEVICES="$GPU"

# ============================================================
#  ARGS + launch
# ============================================================
ARGS=(
    --cfg_file "$CFG_FILE"
    --batch_size "$BATCH_SIZE" --workers "$WORKERS" --epochs "$EPOCHS"
    --extra_tag "$EXTRA_TAG" --output_root "$OUTPUT_ROOT"
)
[ -n "$CKPT" ]                    && ARGS+=(--ckpt "$CKPT")
[ -n "$PRETRAINED_MODEL" ]        && ARGS+=(--pretrained_model "$PRETRAINED_MODEL")
[ "$FIX_RANDOM_SEED" = True ]     && ARGS+=(--fix_random_seed)
[ -n "$LAUNCHER" ]                && ARGS+=(--launcher "$LAUNCHER" --tcp_port "$TCP_PORT" --local_rank "$LOCAL_RANK")
[ "$SYNC_BN" = True ]             && ARGS+=(--sync_bn)
[ -n "$CKPT_SAVE_INTERVAL" ]      && ARGS+=(--ckpt_save_interval "$CKPT_SAVE_INTERVAL")
[ -n "$MAX_CKPT_SAVE_NUM" ]       && ARGS+=(--max_ckpt_save_num "$MAX_CKPT_SAVE_NUM")
[ "$MERGE_ALL_ITERS_TO_ONE_EPOCH" = True ] && ARGS+=(--merge_all_iters_to_one_epoch)
[ -n "$START_EPOCH" ]             && ARGS+=(--start_epoch "$START_EPOCH")
[ -n "$MAX_WAITING_MINS" ]        && ARGS+=(--max_waiting_mins "$MAX_WAITING_MINS")
[ "$SAVE_TO_FILE" = True ]        && ARGS+=(--save_to_file)
[ "$USE_WANDB" = True ]           && ARGS+=(--use_wandb)
[ "$SKIP_EVAL" = True ]           && ARGS+=(--skip_eval)
[ ${#SET_CFGS[@]} -gt 0 ]         && ARGS+=(--set "${SET_CFGS[@]}")

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log"

echo "[train] cfg=$CFG_FILE epochs=$EPOCHS bs=$BATCH_SIZE log=$LOG"
if [ "$RUN_MODE" = "background" ]; then
    nohup python -u tools/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
    disown
    PID=$!
    echo "PID=$PID, log=$LOG"
    echo "跟踪: tail -f $LOG"
else
    python -u tools/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
