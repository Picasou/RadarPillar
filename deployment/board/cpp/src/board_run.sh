#!/usr/bin/env bash
# ⑥ 板上批量验证: points/ 50 帧全链出框 + 分段计时汇总(vs PC golden 口径一致)
# 用法: ./board_run.sh [模型xmodel(默认 models/compiled/dpu_segment.xmodel)]
#       在部署包根目录(fat32/ext4 烧卡后 ext4 分区)执行, weights/ points/ 与本脚本同层
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-models/compiled/dpu_segment.xmodel}"
OUT_DIR="board_out_$(date +%H%M)"
mkdir -p "$OUT_DIR"
N=0
T0=${EPOCHREALTIME:-$(date +%s)}   # busybox date 无 %N, 优先 bash 内建 EPOCHREALTIME
for pts in points/*.bin; do
    ./bin/radar_pipeline_board full "$pts" weights/ "$MODEL" "$OUT_DIR/$(basename "$pts" .bin).txt" \
        || { echo "失败帧: $pts" >&2; exit 1; }
    N=$((N + 1))
done
T1=${EPOCHREALTIME:-$(date +%s)}
# busybox 无 bc: awk 求差与均值
awk -v a="$T0" -v b="$T1" -v n="$N" 'BEGIN{d=b-a; printf "[board_run] %d 帧完成, 总耗时 %.1fs (均 %.3fs/帧)\n", n, d, d/n}'
echo "[board_run] 框文件 → $OUT_DIR/(带回 PC 与 golden 对拍, 口径同 S3/S4)"
