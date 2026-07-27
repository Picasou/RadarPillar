#!/usr/bin/env bash
# baseline_radarpillar 训练 / 复现脚本
# 来源：output/train_log/vod/radarpillar_base/eval_logs/eval_base_train_20260710-182000.log
# 注：原 baseline 训练日志已不可得，本脚本为依据 eval log + cfg 反推的可复现版本

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate angle

cd "$(dirname "$0")/../.."   # 工程根（= 落袋目录的祖上两级）

CFG_FILE="model_store/baseline_radarpillar/cfg.yaml"
EXTRA_TAG="baseline_radarpillar"
BATCH_SIZE_PER_GPU=16
WORKERS=2
EPOCHS=80

# 1) 训练（若已训过只想 val：跳过此步直接 eval）
if [ ! -f "model_store/baseline_radarpillar/best.pth" ]; then
    echo "[train] starting training..."
    CUDA_VISIBLE_DEVICES=0 python -u tools/train.py \
        --cfg_file "$CFG_FILE" \
        --batch_size "$BATCH_SIZE_PER_GPU" \
        --workers "$WORKERS" \
        --epochs "$EPOCHS" \
        --extra_tag "$EXTRA_TAG"
fi

# 2) val (CPU eval，避 numba CUDA 冲突)
echo "[val] running val..."
CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python -u tools/test.py \
    --cfg_file "$CFG_FILE" \
    --batch_size 4 \
    --workers 2 \
    --ckpt "model_store/baseline_radarpillar/best.pth" \
    --extra_tag "${EXTRA_TAG}_val"

echo "[done] baseline_radarpillar 训练/复现完成"
