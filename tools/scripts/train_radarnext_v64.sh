#!/usr/bin/env bash
# 按论文重训：网络严格对齐 RadarNeXt 原工程（VFE/scatter 通道 64，原 OpenPCDet 默认 32 是移植偏离）。
# bs=4 适应 8G 显存（64+bs4 实测峰值 4822MiB）。用户要求网络严格按论文。
set -uo pipefail
CFG_FILE="tools/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen.yaml"
BATCH_SIZE=4
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="rn_mdfen_0717_paper"

cd "$(dirname "$0")/../.."
source /home/admin/anaconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES="$GPU"

ARGS=(
    --cfg_file "$CFG_FILE"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --epochs "$EPOCHS"
    --extra_tag "$EXTRA_TAG"
    --skip_eval --fix_random_seed
    --set OPTIMIZATION.early_stop.enabled False
)

LOG_DIR="output/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen/${EXTRA_TAG}/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d-%H%M%S).log"
echo "log=$LOG"

nohup python -u tools/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID, log=$LOG"
echo "跟踪: tail -f $LOG"
