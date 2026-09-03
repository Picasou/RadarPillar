#!/usr/bin/env bash
# tracker 全链路回灌: centerhead2d_trunk 模型 + full_tracker_pipeline_centertrunk 数据,
# 航迹结果 bin (0200/0201) 落袋回各序列 radar.default/ (overlap=1 覆盖原始对象级输出)。
# 用法: bash tools/scripts/test/replay_tracker_centertrunk.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

# conda 自探测（env=base 本机唯一环境）
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate base

LOG_DIR=".tmp/replay_centertrunk"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/replay_$(date +%Y%m%d-%H%M%S).log"

# 串行: trained(3条) → generalization(8条)
CFGs=(
    "tracker/cfg/cfg_replay_trained_msr_centerhead2d_trunk.yaml"
    "tracker/cfg/cfg_replay_gen_msr_centerhead2d_trunk.yaml"
)

echo "log=$LOG"
for cfg in "${CFGs[@]}"; do
    echo "=== replay: $cfg ===" | tee -a "$LOG"
    python -u -m tracker.tracker --cfg "$cfg" 2>&1 | tee -a "$LOG"
done
echo "=== all done ===" | tee -a "$LOG"
