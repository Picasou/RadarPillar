#!/bin/bash
# unified_rpillar_pipeline.sh — 一脚本端到端: 参数配置 → 环境激活 → train → eval ×N → viz → pickbest → resbag
#
# 用法示例：
#   bash experiments/SH/unified_rpillar_pipeline.sh                        # 默认 rpillar_a4_lnpost 全套
#   SHOW_ARGS=1 bash ...                                                  # 只显示参数, 不跑
#   MODEL=rpillar_a4_rezero bash ...                                      # 切方案 B
#   LAST_N_EVAL=5 RUN_VIZ=false bash ...                                  # 末 5 ckpt 不可视化
#
# 流程图见 doc/RP-unified-pipeline.md (RPiN 阶段 1)

# 注: conda activate 会引入未绑定环境变量(PYTHONPATH/等),严格模式延迟到 conda 激活之后
set -eo pipefail

# ============================================================
#  step 1: 参数配置 (env vars; :="${VAR:=default}" 形式可被 env 覆盖)
# ============================================================
: "${MODEL:=rpillar_a4_lnpost}"
: "${CFG_FILE:=experiments/YAML/a4_lnpost.yaml}"
: "${EPOCHS:=80}"
: "${BATCH_SIZE:=16}"
: "${WORKERS:=2}"
: "${GPU:=0}"
: "${EXTRA_TAG:=${MODEL}}"
: "${DATAROOT:=data/VoD/view_of_delft_PUBLIC/radar_5frames}"
: "${OUTPUT_ROOT:=output/train_log/vod/$(date +%Y%m%d%H%M)_${MODEL}_${EXTRA_TAG}}"
: "${LAST_N_EVAL:=10}"     # 末 N ckpt 跑 eval; 默认 10 (D13 §2.3 漂移 ≥0.1-0.5pp)
: "${RUN_VIZ:=true}"       # eval 后是否跑 visualize_eval.py
: "${RUN_PICKBEST:=true}"  # 是否按 Car R40 max 挑 best.pth
: "${RUN_RESBAG:=true}"    # 是否 resbag 落袋 (artifact 归档)

# ============================================================
#  step 2: 环境激活 (一次性 init; 整个 shell 复用)
# ============================================================
export PYTHONNOUSERSITE=1  # user-local 坏 torch 屏蔽 (memory: torch-user-local-mask)
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/anaconda3" "$HOME/miniconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && { source "$_c/etc/profile.d/conda.sh"; break; }
    done
fi
find_conda_env() {
    local try_envs=("${DESIRED_ENV:-angle}" "angle" "base")
    local installed; installed="$(conda env list 2>/dev/null | awk 'NF && $1 != "#" {print $1}')"
    for env in "${try_envs[@]}"; do
        if echo "$installed" | grep -qx "$env"; then echo "$env"; return 0; fi
    done; return 1
}
TARGET_ENV="$(find_conda_env)" || { echo "[ERROR] 无可用 conda env"; exit 1; }
conda activate "$TARGET_ENV"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[pipeline] conda env=$TARGET_ENV PYTHONNOUSERSITE=$PYTHONNOUSERSITE GPU=$GPU"

# 至此 conda 已激活,所有变量已绑;严格 nounset 模式开启
set -u

# ============================================================
#  step 0.5: dry-run
# ============================================================
if [ "${SHOW_ARGS:-0}" = "1" ]; then
    cat <<EOF
  MODEL        = $MODEL
  CFG_FILE     = $CFG_FILE
  EPOCHS       = $EPOCHS
  BATCH_SIZE   = $BATCH_SIZE
  WORKERS      = $WORKERS
  GPU          = $GPU
  OUTPUT_ROOT  = $OUTPUT_ROOT
  LAST_N_EVAL  = $LAST_N_EVAL
  RUN_VIZ      = $RUN_VIZ
  RUN_PICKBEST = $RUN_PICKBEST
  RUN_RESBAG   = $RUN_RESBAG
EOF
    exit 0
fi

# ============================================================
#  step 3: train (foreground; tee 一份到 log, 上层可 tail)
# ============================================================
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log"
echo "[step 3/6] train start  LOG=$LOG"

python -u tools/train.py \
    --cfg_file "$CFG_FILE" \
    --batch_size "$BATCH_SIZE" --workers "$WORKERS" --epochs "$EPOCHS" \
    --extra_tag "$EXTRA_TAG" --output_root "$OUTPUT_ROOT" \
    --skip_eval \
    --set "OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False" 2>&1 | tee "$LOG"
echo "[step 3/6] train done @ $(date)"

# ============================================================
#  step 4: eval 末 N ckpt (in-process, no separate shell file)
# ============================================================
START_EPOCH=$(( EPOCHS - LAST_N_EVAL ))
echo "[step 4/6] eval 末 ${LAST_N_EVAL} ckpt (start_epoch=$START_EPOCH)"

for ep in $(seq $START_EPOCH $((EPOCHS - 1))); do
    CKPT="${OUTPUT_ROOT}/ckpt/checkpoint_epoch_${ep}.pth"
    [ -f "$CKPT" ] || { echo "[eval] skip ep${ep}: ckpt missing"; continue; }
    echo "[step 4/6] eval ep${ep}  ckpt=$CKPT"
    python -u tools/test.py \
        --cfg_file "$CFG_FILE" --ckpt "$CKPT" \
        --batch_size 4 --workers "$WORKERS" --gpu "$GPU" \
        --extra_tag "${MODEL}_ep${ep}" --eval_tag default \
        --output_root "$OUTPUT_ROOT" 2>&1 | tee "${LOG_DIR}/eval_ep${ep}.log"

    if [ "$RUN_VIZ" = "true" ]; then
        EVAL_DIR=$(ls -td "${OUTPUT_ROOT}/eval/epoch_${ep}/val/default" 2>/dev/null | head -1)
        if [ -n "$EVAL_DIR" ] && [ -d "$EVAL_DIR" ]; then
            echo "[viz] ep${ep} → ${EVAL_DIR}/vis"
            python -u tools/visualize_eval.py \
                --eval_dir "$EVAL_DIR" \
                --dataroot "$DATAROOT" \
                --train_log_dir "$OUTPUT_ROOT" \
                --n_samples 10 --score_thresh 0.1 \
                2>&1 | tee "${LOG_DIR}/viz_ep${ep}.log" || true
        fi
    fi
done

# ============================================================
#  step 5: pickbest (Car R40 max → best.pth)
#  - inline python heredoc; 扫 eval/epoch_*/val/*/ 下 result 找最大 Car_3d_R40 moderate,
#    cp 对应 ckpt 到 best.pth
# ============================================================
if [ "$RUN_PICKBEST" = "true" ]; then
    echo "[step 5/6] pickbest (start_epoch=$START_EPOCH)"
    START_EPOCH="$START_EPOCH" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODEL="$MODEL" \
    python3 <<'PY'
import os, re, glob, shutil, sys
from pathlib import Path
start_epoch = int(os.environ['START_EPOCH'])
out = Path(os.environ['OUTPUT_ROOT'])
eval_root = out / 'eval'
if not eval_root.exists():
    print(f'[pickbest] WARN: {eval_root} 不存在,val 是否跑了？')
    sys.exit(0)

# 扫所有 eval/epoch_*/val/<tag>/ 找 metric
pattern = re.compile(r'Car_3d/moderate_R40[^0-9-]*([0-9.]+)')
results = []  # [(metric, ep, result_file)]
for result in eval_root.rglob('*.json'):
    m = re.search(r'epoch_(\d+)', str(result))
    if not m: continue
    ep = int(m.group(1))
    if ep < start_epoch: continue
    try:
        c = result.read_text(encoding='utf-8', errors='ignore')
        match = pattern.search(c)
        if match:
            results.append((float(match.group(1)), ep, result))
    except Exception:
        pass

if not results:
    print(f'[pickbest] WARN: 没找到 Car_3d/moderate_R40 (>= epoch {start_epoch}), fallback 最新 ckpt')
    ckpt_dir = out / 'ckpt'
    cks = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'),
                 key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
    if not cks:
        print('[pickbest] ERROR: 无任何 ckpt,失败')
        sys.exit(1)
    src = cks[-1]
    ep = int(re.search(r'(\d+)', src.stem).group(1))
else:
    results.sort(key=lambda x: x[0])  # 按 metric 升序,最后一个最大
    metric, ep, _ = results[-1]
    src = out / 'ckpt' / f'checkpoint_epoch_{ep}.pth'
    print(f'[pickbest] best = epoch {ep} (Car_3d/moderate_R40 = {metric:.4f})')

if not src.exists():
    print(f'[pickbest] ERROR: best ckpt 不存在: {src}')
    sys.exit(1)
dst = out / 'best.pth'
shutil.copy2(src, dst)
print(f'[pickbest] best.pth ← {src.name}')
PY
fi

# ============================================================
#  step 6: resbag 落袋 (cfg/ckpt/log/eval/best.pth/last.pth/asset)
# ============================================================
if [ "$RUN_RESBAG" = "true" ]; then
    echo "[step 6/6] resbag"
    python .claude/skills/resbag/scripts/resbag.py make --output_root "$OUTPUT_ROOT" \
        2>&1 | tee "${LOG_DIR}/resbag.log" || true
fi

echo "[pipeline] ALL DONE  OUTPUT_ROOT=$OUTPUT_ROOT  $(date)"
