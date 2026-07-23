#!/bin/bash
# RPiN 前置 1-epoch 验证 .sh — n6（N6=RepDWC+MDFEN 降 bs 8）
# 仿 train_radarpillar.sh 约定；conda activate angle（plan §0.5 S4 实测修正）。
set -e
cd "$(dirname "$0")/../.."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate angle

CFG="experiments/YAML/n6.yaml"
BS=8
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="n6"
OUTPUT_ROOT="output/train_log/vod/$(date +%Y%m%d%H%M)_radarpillar_n6"

ARGS=(
    --cfg_file "$CFG"
    --batch_size "$BS"
    --workers "$WORKERS"
    --epochs "$EPOCHS"
    --extra_tag "$EXTRA_TAG"
    --output_root "$OUTPUT_ROOT"
    --skip_eval
    --set OPTIMIZATION.early_stop.enabled False OPTIMIZATION.LR_WARMUP False
)

LOG="$OUTPUT_ROOT/train_$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT"
echo "[rpin:n6] log=$LOG  bs=$BS  output_root=$OUTPUT_ROOT"
nohup python -u tools/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
