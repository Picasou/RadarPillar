#!/usr/bin/env bash
# msrv2_centerhead2d_trunk_vel best.pth → DATASETv2_SIL 全量纯检测回灌 (原地)
# RUN: mode=1(纯检测) save=1 overlap=1 → 各序列 radar.default/0201.00000.bin 原地覆盖
# 安全: 覆盖前备份原件 → ${DATA_ROOT}_0201bak/<序列>/0201.00000.bin (已备份跳过)
# 幂等: 备份按文件存在跳过; 数据集根 .replay_sil_mode1.done 标记整体跳过
# 自愈: tracker 失败自动重试(共 2 次); 重跑安全(0201 为输出文件, loader 以点云流定帧数)
# 用法: bash tools/scripts/test/replay_tracker_msrv2_sil.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

# conda 自探测 (env=base 本机唯一环境)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
conda activate base

MODEL=msrv2_centerhead2d_trunk_vel
DATA_ROOT=/mnt/d/DataSet/DATASETv2_SIL
BAK_ROOT=/mnt/d/DataSet/DATASETv2_SIL_0201bak
CFG_OUT=tracker/cfg/cfg_sil_${MODEL}_mode1.yaml
LOG_DIR=.tmp/2026-10-13/replay-msrv2-sil
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/replay_sil_$(date +%Y%m%d-%H%M%S).log"

echo "[SIL-replay] start $(date)  model=$MODEL  data=$DATA_ROOT" | tee -a "$LOG"

# ---------- 预检 ----------
[ -d "$DATA_ROOT" ] || { echo "[SIL-replay] FATAL: 数据集不存在 $DATA_ROOT" | tee -a "$LOG"; exit 1; }
nseq=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[SIL-replay] 序列数=$nseq" | tee -a "$LOG"

# 关键输入文件齐全性 (0001 触发/0101 点云/0200+0201 对象流对)
miss=0
while IFS= read -r d; do
    for f in 0001.00000.bin 0101.00000.bin 0200.00000.bin 0201.00000.bin; do
        [ -f "$d/radar.default/$f" ] || { echo "[SIL-replay] 缺文件: $d/radar.default/$f" | tee -a "$LOG"; miss=1; }
    done
done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d)
[ "$miss" -eq 0 ] || { echo "[SIL-replay] FATAL: 存在缺失文件, 中止" | tee -a "$LOG"; exit 1; }

# 模型产物链核验
root="$(ls -1d output/train_log/msr/*_rpillar_${MODEL}_${MODEL} 2>/dev/null | tail -n1)"
if [ -z "$root" ] || [ ! -f "${root}/resbag/best.pth" ] || [ ! -f "${root}/resbag/cfg.yaml" ]; then
    echo "[SIL-replay] FATAL: 模型产物缺失 root=${root:-无}" | tee -a "$LOG"; exit 1
fi
echo "[SIL-replay] ckpt=${root}/resbag/best.pth" | tee -a "$LOG"

# GPU / 磁盘余量 (仅告警不拦截)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
df -h /mnt/d | tail -1 | tee -a "$LOG"

# 整体完成标记 → 跳过
[ -f "${DATA_ROOT}/.replay_sil_mode1.done" ] && { echo "[SIL-replay] 已完成(标记存在), 跳过" | tee -a "$LOG"; exit 0; }

# ---------- 备份原件 (逐文件幂等) ----------
nbak=0
while IFS= read -r d; do
    name="$(basename "$d")"
    mkdir -p "${BAK_ROOT}/${name}"
    if [ ! -f "${BAK_ROOT}/${name}/0201.00000.bin" ]; then
        cp -a "$d/radar.default/0201.00000.bin" "${BAK_ROOT}/${name}/0201.00000.bin" || {
            echo "[SIL-replay] FATAL: 备份失败 ${name}" | tee -a "$LOG"; exit 1; }
        nbak=$((nbak+1))
    fi
done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d)
echo "[SIL-replay] 本次新备份 ${nbak} 份, 备份根=${BAK_ROOT}" | tee -a "$LOG"

# ---------- 生成 cfg + 跑 tracker (重试×2) ----------
python tools/scripts/test/gen_replay_cfg_msrv2.py \
    --base "tracker/cfg/cfg_trained_${MODEL}.yaml" \
    --mode 1 --out "$CFG_OUT" \
    --model-cfg "$(pwd)/${root}/resbag/cfg.yaml" \
    --model-ckpt "$(pwd)/${root}/resbag/best.pth" \
    --seq-root "$DATA_ROOT" 2>&1 | tee -a "$LOG" || { echo "[SIL-replay] FATAL: gen cfg 失败" | tee -a "$LOG"; exit 1; }

ok=1
for attempt in 1 2; do
    echo "[SIL-replay] tracker 尝试 ${attempt}/2 $(date)" | tee -a "$LOG"
    if python -u -m tracker.tracker --cfg "$CFG_OUT" 2>&1 | tee -a "$LOG"; then
        ok=0; break
    fi
    echo "[SIL-replay] tracker 失败(尝试 ${attempt})" | tee -a "$LOG"
done
[ "$ok" -eq 0 ] || { echo "[SIL-replay] FATAL: tracker 两次尝试均失败, 产物不落 done 标记" | tee -a "$LOG"; exit 1; }

# ---------- 收尾核验: 63 序列 0201 均已更新 ----------
nout=$(find "$DATA_ROOT" -mindepth 2 -maxdepth 3 -path "*/radar.default/0201.00000.bin" -newermt "today" | wc -l)
echo "[SIL-replay] 今日更新 0201 计数=${nout}/${nseq}" | tee -a "$LOG"

touch "${DATA_ROOT}/.replay_sil_mode1.done"
echo "[SIL-replay] ALL DONE $(date)" | tee -a "$LOG"
