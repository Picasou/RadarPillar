#!/bin/bash
# After v2 (paper_faithful_rot) finishes, run v1 again (paper_faithful_full).
# Reason: v1 was cut at ep51 by suspend; need a full 80-epoch v1 for fair v1 vs v2 comparison.

set -u

V2_TAG="paper_faithful_rot"
V1_CFG="/home/fatih/Xena_Vision/repo/RadarPillar/tools/cfgs/vod_models/vod_radarpillar.yaml"
V1_TAG="paper_faithful_full_rerun"
V1_LOG="/tmp/rp_full_rerun.log"
CHAIN_LOG="/tmp/rp_chain2.log"

echo "[$(date)] chain2 started, waiting for v2 ($V2_TAG)" >> "$CHAIN_LOG"

PID_V2=$(pgrep -f "extra_tag $V2_TAG" | head -1 || true)
if [ -z "$PID_V2" ]; then
  echo "[$(date)] v2 PID not found, aborting chain2" >> "$CHAIN_LOG"
  exit 1
fi
echo "[$(date)] v2 PID=$PID_V2" >> "$CHAIN_LOG"

while kill -0 "$PID_V2" 2>/dev/null; do
  sleep 60
done
echo "[$(date)] v2 finished" >> "$CHAIN_LOG"

sleep 30

# Extend suspend-inhibit another 4h for v1 rerun (in case original 4h ran out).
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
  --who="rp_train" --why="v1 rerun" --mode=block \
  sleep 14400 > /tmp/rp_inhibit2.log 2>&1 < /dev/null &
disown
echo "[$(date)] suspend-inhibit extended for v1 rerun" >> "$CHAIN_LOG"

echo "[$(date)] starting v1 rerun ($V1_TAG) with $V1_CFG" >> "$CHAIN_LOG"
cd /home/fatih/Xena_Vision/repo/RadarPillar
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file "$V1_CFG" \
  --batch_size 8 \
  --extra_tag "$V1_TAG" \
  --workers 4 \
  >> "$V1_LOG" 2>&1
EXIT=$?
echo "[$(date)] v1 rerun finished, exit=$EXIT" >> "$CHAIN_LOG"
