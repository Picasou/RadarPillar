#!/bin/bash
# RPiN 前置完整训练 .sh（80 epochs）— n1（N1=standard+无neck）
# 注：1-epoch 冒烟验证请用 tools/scripts/rpin_1epoch.py（本 .sh 是全量训练入口）。
# 仿 train_radarpillar.sh 约定；conda activate base（plan §0.5 S4：env=base，angle 为死路径）。
set -e
cd "$(dirname "$0")/../.."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate base

CFG="experiments/YAML/n1.yaml"
BS=16
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="n1"
OUTPUT_ROOT="output/train_log/vod/$(date +%Y%m%d%H%M)_radarpillar_n1"

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
echo "[rpin:n1] log=$LOG  bs=$BS  output_root=$OUTPUT_ROOT"
nohup python -u tools/train.py "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
