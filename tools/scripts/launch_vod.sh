#!/bin/bash
# 启动训练 — 真正脱离 shell session (写到文件执行更稳)
set -e

cd /home/dministrator1/RadarPillar
source /home/dministrator1/miniconda3/etc/profile.d/conda.sh
conda activate angle
export CUDA_VISIBLE_DEVICES=0

EXTRA_TAG="radarpiller_0709"
LOG_DIR="output/cfgs/model/vod_models/vod_radarpillar/${EXTRA_TAG}/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d-%H%M%S).log"

setsid nohup python -u tools/train.py \
    --cfg_file tools/cfgs/model/vod_models/vod_radarpillar.yaml \
    --batch_size 16 \
    --workers 2 \
    --epochs 100 \
    --extra_tag "$EXTRA_TAG" \
    > "$LOG" 2>&1 < /dev/null &

PID=$!
disown
echo "PID=$PID"
echo "LOG=$LOG"
echo "跟踪: tail -f $LOG"
