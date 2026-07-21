#!/bin/bash

#  eg. bash .claude/skills/model-train/eval_vod.sh

set -e

# ════════════════════════════════════════════════════════════════
#  必改
# ════════════════════════════════════════════════════════════════
CFG_FILE="tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml"
BATCH_SIZE=4
WORKERS=2
GPU=0
EXTRA_TAG="rp_base_0716"
DATAROOT="data/VoD/view_of_delft_PUBLIC/radar_5frames"

# 输出根目录 (train.py:57-58 派生 EXP_GROUP_PATH + TAG)
# [output 覆写: 与 train 一致的 OUTPUT_ROOT]
OUTPUT_ROOT="output/train_log/vod/202607171624_radarpiller_bs8"

# ════════════════════════════════════════════════════════════════
#  CPU eval 开关 (val 阶段强制 CPU)
#   - 设为 True: 使用 tools/test_cpu.py (CPU-only test, 跳过 .cuda())
#   - 设为 False: 使用 tools/test.py (GPU 模式)
# ════════════════════════════════════════════════════════════════
CPU_EVAL=True

# ════════════════════════════════════════════════════════════════
#  评估模式
# ════════════════════════════════════════════════════════════════
# single: 评估指定 --ckpt, 训练后用
# all:    评估 ckpt_dir 下所有 checkpoint_epoch_*.pth, 训练中轮询用
# 若 env 已注入 EVAL_MODE（如 autofinish 传 EVAL_MODE=all），尊重之，不覆盖
EVAL_MODE="${EVAL_MODE:-single}"

# ---- single 模式 ----
CKPT="${OUTPUT_ROOT}/ckpt/checkpoint_epoch_80.pth"
EVAL_TAG="default"          # 与 tools/test.py 默认 eval_tag 一致
# SAVE_TO_FILE=True            # 导出 KITTI 格式预测文件

# ---- all 模式 ----
# CKPT_DIR="${OUTPUT_ROOT}/ckpt"
# START_EPOCH=0
# MAX_WAITING_MINS=30

# ════════════════════════════════════════════════════════════════
#  运行模式
# ════════════════════════════════════════════════════════════════
# foreground: 前台 + tee, 终端实时打印, 日志同时落盘
# background: nohup + disown 放后台, 仅打印 PID, 日志在文件
RUN_MODE="foreground"

# ════════════════════════════════════════════════════════════════
#  eval 完可视化 (结构化结果 + loss 曲线 + 多帧 PNG)
# ════════════════════════════════════════════════════════════════
RUN_VIZ=True
N_VIZ_SAMPLES=10                # 可视化帧数 (>=10)
SCORE_THRESH=0.1                # Pred 置信度阈值
TRAIN_LOG_DIR="${OUTPUT_ROOT}"  # 含 logs/ 与 tensorboard/ 的目录


# ════════════════════════════════════════════════════════════════
#  执行
# ════════════════════════════════════════════════════════════════
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

# 注意: 不要 export NUMBA_DISABLE_CUDA=1
# kitti eval 的 rotate_iou_gpu_eval 依赖 numba CUDA, 禁用会直接 CudaSupportError
# (训练期 numba 偶发 SIGSEGV 才需要禁, eval 阶段 numba 路径是稳定的)

# 拼 ARGS
if [ "$EVAL_MODE" = "all" ]; then
    CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/ckpt}"
    ARGS=(
        --cfg_file "$CFG_FILE"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --ckpt_dir "$CKPT_DIR"
        --eval_all
        --extra_tag "$EXTRA_TAG"
    )
    [ -n "$START_EPOCH" ]      && ARGS+=(--start_epoch "$START_EPOCH")
    [ -n "$MAX_WAITING_MINS" ] && ARGS+=(--max_waiting_mins "$MAX_WAITING_MINS")
else
    ARGS=(
        --cfg_file "$CFG_FILE"
        --batch_size "$BATCH_SIZE"
        --workers "$WORKERS"
        --ckpt "$CKPT"
        --extra_tag "$EXTRA_TAG"
        --output_root "${OUTPUT_ROOT}"
    )
    [ -n "$EVAL_TAG" ]          && ARGS+=(--eval_tag "$EVAL_TAG")
    [ "$SAVE_TO_FILE" = True ]  && ARGS+=(--save_to_file)
fi

# 选择 test 脚本 (CPU or GPU)
if [ "$CPU_EVAL" = True ]; then
    TEST_SCRIPT="tools/test_cpu.py"
    unset CUDA_VISIBLE_DEVICES
    export NUMBA_DISABLE_CUDA=1
else
    TEST_SCRIPT="tools/test.py"
fi

LOG_DIR="${OUTPUT_ROOT}/eval_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval_${EVAL_MODE}_$(date +%Y%m%d-%H%M%S).log"
echo "log=$LOG"

# 跑 eval
if [ "$RUN_MODE" = "background" ]; then
    nohup python -u "$TEST_SCRIPT" "${ARGS[@]}" > "$LOG" 2>&1 &
    disown
    PID=$!
    echo "PID=$PID, log=$LOG"
    echo "跟踪: tail -f $LOG"
    wait $PID
else
    python -u "$TEST_SCRIPT" "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
EVAL_EXIT=${PIPESTATUS[0]}

# 跑可视化
if [ "$RUN_VIZ" = True ] && [ "$EVAL_EXIT" = 0 ]; then
    if [ "$EVAL_MODE" = "single" ]; then
        CKPT_BASE=$(basename "$CKPT" .pth)
        if [[ "$CKPT_BASE" =~ checkpoint_epoch_(.+) ]]; then
            EPOCH="${BASH_REMATCH[1]}"
        else
            EPOCH="no_number"
        fi
        EVAL_DIR="${OUTPUT_ROOT}/eval/epoch_${EPOCH}/val/${EVAL_TAG:-val_eval}"
    else
        EVAL_DIR=$(ls -td "${OUTPUT_ROOT}"/eval/epoch_*/val/*/ 2>/dev/null | head -1)
        EVAL_DIR="${EVAL_DIR%/}"
    fi

    if [ -d "$EVAL_DIR" ]; then
        VIZ_LOG="$LOG_DIR/viz_$(date +%Y%m%d-%H%M%S).log"
        echo ""
        echo "Running visualize_eval.py -> ${EVAL_DIR}/vis/"
        python -u tools/visualize_eval.py \
            --eval_dir "$EVAL_DIR" \
            --dataroot "$DATAROOT" \
            --train_log_dir "$TRAIN_LOG_DIR" \
            --n_samples "$N_VIZ_SAMPLES" \
            --score_thresh "$SCORE_THRESH" \
            2>&1 | tee "$VIZ_LOG" || true
    else
        echo "[warn] EVAL_DIR not found: $EVAL_DIR, skip visualize_eval.py"
    fi
fi
