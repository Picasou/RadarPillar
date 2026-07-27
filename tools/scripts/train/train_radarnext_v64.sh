#!/usr/bin/env bash
# 按论文重训：网络严格对齐 RadarNeXt 原工程（VFE/scatter 通道 64，原 OpenPCDet 默认 32 是移植偏离）。
# bs / workers 由 bs×workers sweep 测定：bs=8 / w=2 在 16GB 卡上 peak 14.95 sps。
set -uo pipefail
CFG_FILE="tools/cfgs/model/vod_models/radarnext/vod_radarnext_mdfen.yaml"
BATCH_SIZE=4
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="rn_mdfen_0717_paper"

cd "$(dirname "$0")/../.."
# 本机真实 conda：/home/admin/anaconda3 + env base（dministrator1/angle 是死路径，参见 memory/env-conda-base.md）
source /home/admin/anaconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES="$GPU"
# sweep 脚本约定：PYTHONPATH=tools（参见 memory/radarpillar-env-ground-truth.md）
export PYTHONPATH="$(pwd)/tools:${PYTHONPATH:-}"

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
