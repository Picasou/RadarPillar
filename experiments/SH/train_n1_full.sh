#!/bin/bash
# experiments/SH/train_n1_full.sh — n1 完整 pipeline (train→eval→pickbest→resbag)
#
# 协议: bs=8 epochs=80, 末 10 ckpt eval, pickbest=Car_3d/moderate_R40 median
# 输出: output/train_log/vod/<ts>_rpillar_n1_n1/  (动态时间戳)
# Marker: output/n1.done  (稳定, 给 model-train skill 的 workflow oracle 判 skip)
#
# 触发: 由 .claude/skills/model-train skill 的 workflow 串行调用
# 与旧 b8 bs16 协议对齐: 标准 PipelineBackbone(非 RepDWC)做 standard 公平对照

set -uo pipefail

cd "$(dirname "$0")/../.."   # 工程根
export PYTHONNOUSERSITE=1

# === conda ===
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
find_conda_env() {
    local try_envs=("${DESIRED_ENV:-base}" "base")
    local installed; installed="$(conda env list 2>/dev/null | awk 'NF && $1 != "#" {print $1}')"
    for env in "${try_envs[@]}"; do
        if echo "$installed" | grep -qx "$env"; then echo "$env"; return 0; fi
    done; return 1
}
conda activate "$(find_conda_env)"

# === 配置 ===
TAG=n1
MODEL=rpillar_${TAG}
CFG=experiments/YAML/${TAG}.yaml
EPOCHS=80
BS=8
GPU=0
WORKERS=2
LAST_N=10
DATAROOT=data/VoD/view_of_delft_PUBLIC/radar_5frames

TS=$(date +%Y%m%d%H%M)
OUTPUT_ROOT=output/train_log/vod/${TS}_${MODEL}_${TAG}
LOG_DIR=${OUTPUT_ROOT}/logs
LOG=${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log
MARKER=output/${TAG}.done

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$LOG_DIR"

echo "[n1] start  ts=$TS  bs=$BS  ep=$EPOCHS  OUTPUT_ROOT=$OUTPUT_ROOT"

# === step 1: train (--skip_eval) ===
python -u tools/train.py \
    --cfg_file "$CFG" \
    --batch_size "$BS" --workers "$WORKERS" --epochs "$EPOCHS" \
    --extra_tag "$TAG" --output_root "$OUTPUT_ROOT" \
    --skip_eval \
    --set "OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False" 2>&1 | tee "$LOG"

# NaN/inf 守卫
if grep -aiE "loss=nan|loss=inf" "$LOG" | tail -5 | grep -qaiE "nan|inf"; then
    echo "[n1] FATAL: train log 含 nan/inf, 中止 (不落 marker, 触发 retry)"; exit 1
fi

# === step 2: eval 末 N ckpt ===
START_EPOCH=$(( EPOCHS - LAST_N ))
for ep in $(seq $START_EPOCH $((EPOCHS - 1))); do
    CKPT="${OUTPUT_ROOT}/ckpt/checkpoint_epoch_${ep}.pth"
    [ -f "$CKPT" ] || { echo "[n1] skip ep${ep} (ckpt 不在)"; continue; }
    python -u tools/test.py \
        --cfg_file "$CFG" --ckpt "$CKPT" \
        --batch_size 4 --workers "$WORKERS" \
        --extra_tag "${MODEL}_ep${ep}" --eval_tag default \
        --output_root "$OUTPUT_ROOT" 2>&1 | tee "${LOG_DIR}/eval_ep${ep}.log" || true
done

# === step 3: pickbest (Car_3d/moderate_R40 median) ===
START_EPOCH="$START_EPOCH" OUTPUT_ROOT="$OUTPUT_ROOT" python3 <<'PY'
import os, re, shutil
from pathlib import Path
start_epoch = int(os.environ['START_EPOCH'])
out = Path(os.environ['OUTPUT_ROOT'])
eval_root = out / 'eval'
pattern = re.compile(r'Car_3d/moderate_R40[^0-9-]*([0-9.]+)')
results = []
if eval_root.exists():
    for r in eval_root.rglob('*.json'):
        m = re.search(r'epoch_(\d+)', str(r))
        if not m: continue
        ep = int(m.group(1))
        if ep < start_epoch: continue
        try:
            c = r.read_text(encoding='utf-8', errors='ignore')
            m2 = pattern.search(c)
            if m2: results.append((float(m2.group(1)), ep, r))
        except Exception: pass
if not results:
    cks = sorted((out / 'ckpt').glob('checkpoint_epoch_*.pth'),
                 key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
    if not cks:
        print('[pickbest] ERROR: 无 ckpt'); raise SystemExit(1)
    src = cks[-1]
else:
    results.sort(key=lambda x: x[0])
    _, ep, _ = results[len(results)//2]
    src = out / 'ckpt' / f'checkpoint_epoch_{ep}.pth'
if not src.exists():
    print(f'[pickbest] ERROR: {src} 不存在'); raise SystemExit(1)
shutil.copy2(src, out / 'best.pth')
print(f'[pickbest] best.pth ← {src.name}')
PY

[ -f "${OUTPUT_ROOT}/best.pth" ] || { echo "[n1] ERROR: best.pth 未生成"; exit 1; }

# === step 4: resbag 落袋 ===
python .claude/skills/resbag/resbag.py make \
    --output_root "$OUTPUT_ROOT" --dataset vod \
    --tag "$TAG" --model "$MODEL" \
    --cfg_file "$CFG" --batch_size "$BS" 2>&1 | tee "${LOG_DIR}/resbag.log"

[ -f "${OUTPUT_ROOT}/model_store.yaml" ] || { echo "[n1] ERROR: model_store.yaml 未落盘"; exit 1; }

# === step 5: 落 marker (workflow oracle 判 idempotent) ===
touch "$MARKER"
echo "[n1] ALL DONE  $OUTPUT_ROOT  marker=$MARKER  $(date)"
