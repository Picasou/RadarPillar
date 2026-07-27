#!/bin/bash

#  eg. bash tools/scripts/eval_vod.sh

set -e

# ════════════════════════════════════════════════════════════════
#  必改
# ════════════════════════════════════════════════════════════════
CFG_FILE="tools/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen.yaml"
BATCH_SIZE=4                # eval 显存裕度大，固定小 bs 即可
WORKERS=2                   # Task 2.5 sweep: RN_MDFEN_W
GPU=0
EXTRA_TAG="rn_mdfen_0716"
# 原模板的 /mnt/d/DATASET/VoD/.../radar_5frames 为错误路径
# （缺 extracted/ 层 + 多嵌套一层；9p/NTFS 挂载大小写不敏感，DATASET 本身可用）。
# 统一用 repo 软链 data/VoD/...（→ /mnt/d/DataSet/vod/extracted/view_of_delft_PUBLIC/radar_5frames），
# 与 yaml DATA_PATH 一致，eval/visualize 的 --dataroot 才能命中真实数据。
DATAROOT="data/VoD/view_of_delft_PUBLIC/radar_5frames"

# 输出根目录：必须与 train.py 实际产物路径一致。
# OpenPCDet 按 CFG_FILE 的相对路径在 output/ 下镜像建目录，
# CFG_FILE 含 radarnext/ 层 → 产物落在 .../vod_models/radarnext/vod_radarnext_mdfen/${EXTRA_TAG}
# （原注释误判为路径压平，少写 radarnext/ 层会致 CKPT 选取 fatal exit）
OUTPUT_ROOT="output/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen/${EXTRA_TAG}"

# CKPT 动态选取（early_stop 禁用后无 checkpoint_best.pth；崩溃/早停时 epoch_80 可能不存在）
if [ -f "${OUTPUT_ROOT}/ckpt/checkpoint_best.pth" ]; then
    CKPT="${OUTPUT_ROOT}/ckpt/checkpoint_best.pth"
else
    CKPT="$(ls -t ${OUTPUT_ROOT}/ckpt/checkpoint_epoch_*.pth 2>/dev/null | head -1)"
fi
# eval 调用前 guard
[ -f "$CKPT" ] || { echo "[FATAL] ckpt missing: $CKPT"; ls -la ${OUTPUT_ROOT}/ckpt/ 2>/dev/null; exit 1; }

# ════════════════════════════════════════════════════════════════
#  评估模式
# ════════════════════════════════════════════════════════════════
# single: 评估指定 --ckpt, 训练后用
# all:    评估 ckpt_dir 下所有 checkpoint_epoch_*.pth, 训练中轮询用
EVAL_MODE="single"

# ---- single 模式 ----
# CKPT 由上方「必改」块的动态选取逻辑赋值（best 优先，否则最新 epoch），此处不再硬编码
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
cd "$(dirname "$0")/../.."
source /home/admin/anaconda3/etc/profile.d/conda.sh
conda activate base
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
    )
    [ -n "$EVAL_TAG" ]          && ARGS+=(--eval_tag "$EVAL_TAG")
    [ "$SAVE_TO_FILE" = True ]  && ARGS+=(--save_to_file)
fi

LOG_DIR="${OUTPUT_ROOT}/eval_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval_${EVAL_MODE}_$(date +%Y%m%d-%H%M%S).log"
echo "log=$LOG"

# 跑 eval
if [ "$RUN_MODE" = "background" ]; then
    nohup python -u tools/test.py "${ARGS[@]}" > "$LOG" 2>&1 &
    disown
    PID=$!
    echo "PID=$PID, log=$LOG"
    echo "跟踪: tail -f $LOG"
    wait $PID
else
    python -u tools/test.py "${ARGS[@]}" 2>&1 | tee "$LOG"
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
