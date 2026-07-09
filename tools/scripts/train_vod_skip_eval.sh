#!/bin/bash
# 100% 不停版本：跳过训练期间 eval + 训练后自动评估也跳过
# 训练完后手动跑 test_eval

CFG_FILE="tools/cfgs/model/vod_models/vod_radarpillar.yaml"
BATCH_SIZE=16
WORKERS=2
EPOCHS=100
GPU=0
EXTRA_TAG="radarpiller_0709"

cd "$(dirname "$0")/../.."
source /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
export CUDA_VISIBLE_DEVICES="$GPU"

LOG_DIR="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA_TAG}/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_skip_eval_$(date +%Y%m%d-%H%M%S).log"

nohup python -u tools/train.py \
    --cfg_file "$CFG_FILE" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --epochs "$EPOCHS" \
    --extra_tag "$EXTRA_TAG" \
    --skip_eval \
    > "$LOG" 2>&1 &
disown
PID=$!
echo "PID=$PID, log=$LOG"
echo "跟踪: tail -f $LOG"
