#!/bin/bash

#  eg. bash tools/scripts/train_vod.sh

# —— 可选 ——
# [续训]
# CKPT="output/.../ckpt/checkpoint_epoch_12.pth"

# [预训练]
# PRETRAINED_MODEL="path/to/pretrained.pth"

# [固定种子]
# FIX_RANDOM_SEED=True

# [分布式]
# LAUNCHER="pytorch"
# TCP_PORT=18888
# LOCAL_RANK=0
# SYNC_BN=True

# [ckpt 保存间隔]
# CKPT_SAVE_INTERVAL=1

# [最多 ckpt 数]
# MAX_CKPT_SAVE_NUM=30

# [iter 合并 1 epoch]
# MERGE_ALL_ITERS_TO_ONE_EPOCH=True

# [起始 epoch]
# START_EPOCH=0

# [数据加载超时(分)]
# MAX_WAITING_MINS=0

# [保存 metric]
# SAVE_TO_FILE=True

# [wandb]
# USE_WANDB=True

# [跳过评估]
SKIP_EVAL=True

# [运行模式]
# foreground: 前台运行 + tee，终端实时打印，日志同时落盘
# background: nohup + disown 放后台，仅打印 PID，日志在文件
RUN_MODE="background"

# [关掉训练期 eval] — early_stop.enabled=False → eval_loader=None → 训练期不 eval
# [no warmup] — 对齐 0709 reference (LR_WARMUP=False)
SET_CFGS=("OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False")

# [cfg 覆盖]
# —— 必改 ——
CFG_FILE="tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml"
BATCH_SIZE=16
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="rp_base_0716"

# [output 覆写: 让 train/test 直接写到 output/train_log/vod/<datetime>_rp_base_0716_paper-bs8/]
OUTPUT_ROOT="output/train_log/vod/$(date +%Y%m%d%H%M)_radarpiller_bs8"


# ============================================================
# train.py 自适应脚本
# ============================================================
cd "$(dirname "$0")/../../.."
# conda 自探测（不写死 /home/xxx），env=angle
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate angle
export CUDA_VISIBLE_DEVICES="$GPU"

ARGS=(
    --cfg_file "$CFG_FILE"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --epochs "$EPOCHS"
    --extra_tag "$EXTRA_TAG"
    --output_root "$OUTPUT_ROOT"
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
LOG="$LOG_DIR/train_$(date +%Y%m%d-%H%M%S).log"

# 前台运行 + tee：终端实时打印，同时落盘到日志文件
echo "log=$LOG"
if [ "$RUN_MODE" = "background" ]; then
    nohup python -u tools/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
    disown
    PID=$!
    echo "PID=$PID, log=$LOG"
    echo "跟踪: tail -f $LOG"
else
    python -u tools/train.py "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
