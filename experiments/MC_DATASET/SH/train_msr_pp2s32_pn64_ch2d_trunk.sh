#!/bin/bash

# eg. bash experiments/MC_DATASET/SH/train_msr_pp2s32_pn64_ch2d_trunk.sh
# MSR (MC_Single_Radar) RadarPillar 行38 BaseBEVBackbone 两级(3,5)等通道(32,32) + PN[64] + ch2d trunk head 变体
# 数据: /mnt/d/DataSet/MSRv1  | 类别: ['1','2','4','5'] (1=轿车 2=行人 4=二轮车 5=卡车)

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

# [跳过评估] — MSR 无独立 evaluation 方法,必须 SKIP
SKIP_EVAL=True

# [运行模式]
# foreground: 前台运行 + tee,终端实时打印,日志同时落盘
# background: nohup + disown 放后台,仅打印 PID,日志在文件
RUN_MODE="background"

# [关掉训练期 eval] — early_stop.enabled=False → eval_loader=None → 训练期不 eval
# [no warmup] — 对齐 reference
SET_CFGS=("OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False")

# —— 必改 ——
CFG_FILE="experiments/MC_DATASET/YAML/msr_pp2s32_pn64_ch2d_trunk.yaml"
BATCH_SIZE=8          # MSR 点云密集 + 8G 显存,保守起步;OOM 则降到 2
WORKERS=4
EPOCHS=80
GPU=0
EXTRA_TAG="msr_pp2s32_pn64_ch2d_trunk"

# [output 覆写: 让 train/test 直接写到 output/train_log/msr/<datetime>_msr_baseline/]
OUTPUT_ROOT="output/train_log/msr/$(date +%Y%m%d%H%M)_${EXTRA_TAG}"


# ============================================================
# train.py 自适应脚本
# 脚本在 experiments/MC_DATASET/SH/,到仓库根是 2 级 ../..
# ============================================================
cd "$(dirname "$0")/../.."

# conda 自探测(不写死 /home/xxx)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi

# conda env fallback helper — 探测顺序: ${DESIRED_ENV:-angle} -> angle -> base
find_conda_env() {
    local try_envs=("${DESIRED_ENV:-angle}" "angle" "base")
    local installed
    installed="$(conda env list 2>/dev/null | awk 'NF && $1 != "#" {print $1}')"
    for env in "${try_envs[@]}"; do
        if echo "$installed" | grep -qx "$env"; then
            echo "$env"; return 0
        fi
    done
    return 1
}
TARGET_ENV="$(find_conda_env)" || {
    echo "[ERROR] 无可用 conda env (尝试过: ${DESIRED_ENV:-angle} -> angle -> base)"
    echo "[ERROR] 请先创建 env: conda create -n angle python=3.x && conda activate angle && pip install -r requirements.txt"
    exit 1
}
echo "[train] 使用 conda env: $TARGET_ENV"
conda activate "$TARGET_ENV"

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
