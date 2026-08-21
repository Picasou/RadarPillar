#!/usr/bin/env bash
# MSR test.py 评估入口(MsrDataset.evaluation 落地后可用)。
# 用法: ./eval_msr.sh [run_dir(默认 rpillar_msr 首跑)] [ckpt(默认 best.pth)]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

RUN_DIR="${1:-output/train_log/msr/202608181912_rpillar_msr_msr}"
CKPT="${2:-best.pth}"

PYTHONPATH=tools python tools/test.py \
    --cfg_file "${RUN_DIR}/msr.yaml" \
    --ckpt "${RUN_DIR}/${CKPT}" \
    --batch_size 4 \
    --workers 2 \
    --output_root "${RUN_DIR}" \
    --extra_tag "eval_${CKPT%.pth}"
