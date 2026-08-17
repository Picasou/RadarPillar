#!/bin/bash
# __FULL_FILENAME__ — __TAG__ 完整链 (train→eval→pickbest[max+median]→resbag)
# 自动生成 by model-train skill (模板: templates/full_chain.template.sh)
# 勿手改 — 改模板, 重生成

set -uo pipefail

# 工程根 = skill 路径推 (.tmp/model-train/<task>/ 上 4 级)
cd "$(dirname "$0")/../../.."   # .tmp/model-train/<task>/ -> 仓库根
export PYTHONNOUSERSITE=1

# === conda 自探测 ===
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
        echo "$installed" | grep -qx "$env" && { echo "$env"; return 0; }
    done; return 1
}
conda activate "$(find_conda_env)" || { echo "[__TAG__] conda env 激活失败"; exit 1; }

# === 配置 (env 覆盖) ===
TAG=__TAG__
MODEL=__MODEL__
CFG=__CFG__
EPOCHS=${EPOCHS:-80}
BS=${BS:-8}
GPU=${GPU:-0}
WORKERS=${WORKERS:-2}
LAST_N=${LAST_N:-10}
DATAROOT=${DATAROOT:-data/VoD/view_of_delft_PUBLIC/radar_5frames}

# H6: OUTPUT_ROOT 首次生成记入 output/<TAG>.root; retry/watchdog 重启时复用旧 root
#     → train.py 同 root auto-resume 生效, 中断不再整链重训。
#     强制全新训练: FRESH=1 (或删 output/<TAG>.root)
ROOT_FILE="output/${TAG}.root"
if [ -n "${FRESH:-}" ]; then
    rm -f "$ROOT_FILE"
fi
if [ -z "${OUTPUT_ROOT:-}" ] && [ -f "$ROOT_FILE" ]; then
    SAVED_ROOT="$(cat "$ROOT_FILE")"
    [ -d "$SAVED_ROOT" ] && OUTPUT_ROOT="$SAVED_ROOT"
fi
if [ -z "${OUTPUT_ROOT:-}" ]; then
    TS=$(date +%Y%m%d%H%M)
    OUTPUT_ROOT="output/train_log/vod/${TS}_${MODEL}_${TAG}"
fi
echo "$OUTPUT_ROOT" > "$ROOT_FILE"
LOG_DIR=${OUTPUT_ROOT}/logs
LOG=${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log
MARKER=output/${TAG}.done

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$LOG_DIR"

echo "[__TAG__] start  ts=$TS  bs=$BS  ep=$EPOCHS  OUTPUT_ROOT=$OUTPUT_ROOT"

# === step 1: train (--skip_eval, 训后补 eval) ===
python -u tools/train.py \
    --cfg_file "$CFG" \
    --batch_size "$BS" --workers "$WORKERS" --epochs "$EPOCHS" \
    --extra_tag "$TAG" --output_root "$OUTPUT_ROOT" \
    --skip_eval \
    --set "OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False" 2>&1 | tee "$LOG"

# NaN/inf 守卫
if grep -aiE "loss=nan|loss=inf" "$LOG" | tail -5 | grep -qaiE "nan|inf"; then
    echo "[__TAG__] FATAL: train log 含 nan/inf, 中止 (不落 marker, 触发 retry)"; exit 1
fi

# === step 2: eval 末 N ckpt ===
START_EPOCH=$(( EPOCHS - LAST_N ))
for ep in $(seq $START_EPOCH $((EPOCHS - 1))); do
    CKPT="${OUTPUT_ROOT}/ckpt/checkpoint_epoch_${ep}.pth"
    [ -f "$CKPT" ] || { echo "[__TAG__] skip ep${ep} (ckpt 不在)"; continue; }
    python -u tools/test.py \
        --cfg_file "$CFG" --ckpt "$CKPT" \
        --batch_size 4 --workers "$WORKERS" \
        --extra_tag "${MODEL}_ep${ep}" --eval_tag default \
        --output_root "$OUTPUT_ROOT" 2>&1 | tee "${LOG_DIR}/eval_ep${ep}.log" || true
done

# === step 3: pickbest (max + median 双落) ===
START_EPOCH="$START_EPOCH" OUTPUT_ROOT="$OUTPUT_ROOT" python3 <<'PY'
import os, re, shutil
from pathlib import Path
start_epoch = int(os.environ['START_EPOCH'])
out = Path(os.environ['OUTPUT_ROOT'])
pattern = re.compile(r'Car_3d/moderate_R40[^0-9-]*([0-9.]+)')
results = []
for r in (out / 'eval').rglob('*.json'):
    m = re.search(r'epoch_(\d+)', str(r))
    if not m: continue
    ep = int(m.group(1))
    if ep < start_epoch: continue
    try:
        c = r.read_text(encoding='utf-8', errors='ignore')
        m2 = pattern.search(c)
        if m2: results.append((float(m2.group(1)), ep))
    except Exception: pass

if not results:
    cks = sorted((out / 'ckpt').glob('checkpoint_epoch_*.pth'),
                 key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
    if not cks:
        print('[pickbest] ERROR: 无 ckpt'); raise SystemExit(1)
    shutil.copy2(cks[-1], out / 'best.pth')
    shutil.copy2(cks[-1], out / 'best_median.pth')
    print(f'[pickbest] 无 metric, best/best_median ← {cks[-1].name} (fallback 最新)')
else:
    results.sort(key=lambda x: x[0])
    max_metric, max_ep = results[-1]
    med_metric, med_ep = results[len(results) // 2]
    max_src = out / 'ckpt' / f'checkpoint_epoch_{max_ep}.pth'
    med_src = out / 'ckpt' / f'checkpoint_epoch_{med_ep}.pth'
    for src_name, ep, metric, dst in [(max_src, max_ep, max_metric, 'best.pth'),
                                       (med_src, med_ep, med_metric, 'best_median.pth')]:
        if not src_name.exists():
            print(f'[pickbest] ERROR: {src_name} 不存在'); raise SystemExit(1)
        shutil.copy2(src_name, out / dst)
    print(f'[pickbest] best.pth(max) ← ep{max_ep} ({max_metric:.4f}), '
          f'best_median.pth(median) ← ep{med_ep} ({med_metric:.4f})')
PY

[ -f "${OUTPUT_ROOT}/best.pth" ] || { echo "[__TAG__] ERROR: best.pth 未生成"; exit 1; }

# === step 4: resbag 落袋 ===
python .claude/skills/resbag/resbag.py make \
    --output_root "$OUTPUT_ROOT" --dataset vod \
    --tag "$TAG" --model "$MODEL" \
    --cfg_file "$CFG" --batch_size "$BS" 2>&1 | tee "${LOG_DIR}/resbag.log"

[ -f "${OUTPUT_ROOT}/model_store.yaml" ] || { echo "[__TAG__] ERROR: model_store.yaml 未落盘"; exit 1; }

# === step 5: 落 marker (workflow oracle 判 idempotent) ===
touch "$MARKER"
echo "[__TAG__] ALL DONE  $OUTPUT_ROOT  marker=$MARKER  $(date)"
