#!/bin/bash
# Multi-seed paper_faithful_rot (v2 config) — 3 sequential runs.
# Goal: establish mean +/- std AP and pick best seed.
#
# Detached: setsid + nohup + disown.
# Survives lid close: systemd-inhibit blocks suspend for 8h.
# Each run: ~2.5h. Total: ~7.5h.

set -u

CFG="/home/fatih/Xena_Vision/repo/RadarPillar/tools/cfgs/vod_models/vod_radarpillar_rot.yaml"
TAG_PREFIX="paper_faithful_rot_s"
CHAIN_LOG="/home/fatih/Xena_Vision/repo/RadarPillar/experiments/logs/multiseed_chain.log"
LOG_DIR="/home/fatih/Xena_Vision/repo/RadarPillar/experiments/logs"

mkdir -p "$LOG_DIR"

echo "[$(date)] multi-seed chain started" >> "$CHAIN_LOG"

# Renew suspend-inhibit (8h covers all 3 runs).
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
  --who="rp_multiseed" --why="3-seed v2 training" --mode=block \
  sleep 28800 > /dev/null 2>&1 < /dev/null &
disown
echo "[$(date)] suspend-inhibit armed for 8h" >> "$CHAIN_LOG"

for SEED in 1 2 3; do
  TAG="${TAG_PREFIX}${SEED}"
  LOG_FILE="${LOG_DIR}/${TAG}.log"
  echo "[$(date)] starting run $TAG (seed via FIX_RANDOM_SEED=False random init)" >> "$CHAIN_LOG"

  cd /home/fatih/Xena_Vision/repo/RadarPillar
  CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file "$CFG" \
    --batch_size 8 \
    --extra_tag "$TAG" \
    --workers 4 \
    >> "$LOG_FILE" 2>&1
  EXIT=$?
  echo "[$(date)] $TAG finished, exit=$EXIT" >> "$CHAIN_LOG"

  # Small buffer between runs (let GPU/RAM settle).
  sleep 30
done

echo "[$(date)] multi-seed chain complete" >> "$CHAIN_LOG"
