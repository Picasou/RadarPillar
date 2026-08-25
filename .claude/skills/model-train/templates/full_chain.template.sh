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
# 默认每个 split（train + val）生成 10 帧 BEV 可视化 = 20 张总图
VIZ_NUM=${VIZ_NUM:-10}
# seed 固定 (train.py --fix_random_seed → set_random_seed(42), 压过 cfg 的 FIX_RANDOM_SEED:false); FIX_SEED=False 关闭
FIX_SEED=${FIX_SEED:-True}

# 数据集感知: 从 cfg _BASE_CONFIG_ 推 DS → OUTPUT_ROOT 子目录 / resbag --dataset / eval 跳过 / viz 分派
DS_YAML=$(python3 - "$CFG" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(r'_BASE_CONFIG_:\s*(\S+)', txt)
print(m.group(1) if m else '')
PY
)
case "$DS_YAML" in
    *msr_dataset.yaml*) DS=msr ;;
    *nuscenes*)         DS=nuscenes ;;
    *)                  DS=vod ;;
esac
echo "[__TAG__] dataset=${DS} (base=${DS_YAML:-无})"

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
    OUTPUT_ROOT="output/train_log/${DS}/${TS}_${MODEL}_${TAG}"
fi
echo "$OUTPUT_ROOT" > "$ROOT_FILE"
LOG_DIR=${OUTPUT_ROOT}/logs
LOG=${LOG_DIR}/train_$(date +%Y%m%d-%H%M%S).log
MARKER=output/${TAG}.done

export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$LOG_DIR"

echo "[__TAG__] start  ts=${TS:-?}  bs=$BS  ep=$EPOCHS  OUTPUT_ROOT=$OUTPUT_ROOT"

# === step 1: train (--skip_eval, 训后补 eval) ===
# OOM 回退: bs>4 时 OOM → 清 ckpt 以 bs4 全量重训 (record BS_EFF 供汇报/落袋)
SEED_ARGS=(); [ "$FIX_SEED" = True ] && SEED_ARGS=(--fix_random_seed)
run_train() {
    python -u tools/train.py \
        --cfg_file "$CFG" \
        --batch_size "$1" --workers "$WORKERS" --epochs "$EPOCHS" \
        --extra_tag "$TAG" --output_root "$OUTPUT_ROOT" \
        --skip_eval "${SEED_ARGS[@]}" \
        --set "OPTIMIZATION.early_stop.enabled" "False" "OPTIMIZATION.LR_WARMUP" "False" 2>&1 | tee "$LOG"
}
BS_EFF="$BS"
echo "[__TAG__] train attempt bs=$BS"
run_train "$BS"; RC=$?
if [ "$RC" -ne 0 ] && grep -aiE "out of memory|cuda error" "$LOG" | grep -qiE "out of memory|cuda error"; then
    if [ "$BS" -gt 4 ]; then
        echo "[__TAG__] OOM @bs${BS} → 清 ckpt, 回退 bs4 全量重训"
        BS_EFF=4
        rm -rf "${OUTPUT_ROOT}/ckpt"
        run_train 4; RC=$?
    else
        echo "[__TAG__] OOM @bs${BS} (已最低), 中止 (触发 retry)"; exit 1
    fi
fi
[ "$RC" -ne 0 ] && { echo "[__TAG__] train 失败 (rc=$RC)"; exit 1; }
echo "$BS_EFF" > "${OUTPUT_ROOT}/.bs_effective"
echo "[__TAG__] train done, BS_EFF=$BS_EFF"

# NaN/inf 守卫
if grep -aiE "loss=nan|loss=inf" "$LOG" | tail -5 | grep -qaiE "nan|inf"; then
    echo "[__TAG__] FATAL: train log 含 nan/inf, 中止 (不落 marker, 触发 retry)"; exit 1
fi

# === step 2: eval 末 N ckpt (MSR 亦有 evaluation, 全 DS 统一跑, 供 BEV pickbest) ===
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

# === step 3: pickbest (max + median 双落; 口径=2D/BEV mean, 2DNoZ 系 3D AP 恒 0 勿用) ===
START_EPOCH="$START_EPOCH" OUTPUT_ROOT="$OUTPUT_ROOT" python3 <<'PY'
import os, re, shutil
from pathlib import Path
start_epoch = int(os.environ['START_EPOCH'])
out = Path(os.environ['OUTPUT_ROOT'])
BEV_RE = re.compile(r'^[A-Za-z]+_bev/moderate_R40$')
R3D_RE = re.compile(r'^[A-Za-z]+_3d/moderate_R40$')

def bev_mean(ret):
    """2D 口径: 全部 *_bev/moderate_R40 均值 (4 类齐全即 2D mAP)。无 bev → None。"""
    vals = [v for k, v in ret.items()
            if BEV_RE.match(k) and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None

def car3d(ret):
    for k, v in ret.items():
        if k == 'Car_3d/moderate_R40' and isinstance(v, (int, float)):
            return v
    return None

results = []
for r in (out / 'eval').rglob('results.json'):
    m = re.search(r'epoch_(\d+)', str(r))
    if not m: continue
    ep = int(m.group(1))
    if ep < start_epoch: continue
    try:
        import json
        ret = json.loads(r.read_text(encoding='utf-8', errors='ignore')).get('ret_dict', {})
        score = bev_mean(ret)
        if score is None:
            score = car3d(ret)          # 兜底: 无 bev 键时退 3D Car
        if score is not None:
            results.append((float(score), ep))
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

# === step 4: viz (best.pth 对 train/val 各 VIZ_NUM 帧, GT 实线+pred 虚线) ===
# 按 DS 分派 viz 脚本 (推导见配置区);图落 OUTPUT_ROOT/viz/, 随 resbag 归档。
# 默认 VIZ_NUM=10 × VIZ_SPLITS=both(train+val) = 20 张总图
VIZ_NUM=${VIZ_NUM:-10}
case "$DS" in
    msr)      VIZ_SCRIPT="tools/utils/visual_utils/visualize_msr.py" ;;
    nuscenes) VIZ_SCRIPT="tools/utils/visual_utils/visualize_nuscenes.py" ;;
    vod)      VIZ_SCRIPT="tools/utils/visual_utils/visualize_kitti.py" ;;
    *) echo "[__TAG__] WARN: 未知数据集 ${DS}, 跳过 viz"; VIZ_SCRIPT="" ;;
esac
VIZ_SPLITS="both"
if [ -n "$VIZ_SCRIPT" ]; then
    echo "[__TAG__] viz: $VIZ_SCRIPT --split $VIZ_SPLITS --num $VIZ_NUM -> ${OUTPUT_ROOT}/viz/"
    python -u "$VIZ_SCRIPT" \
        --cfg_file "$CFG" --ckpt "${OUTPUT_ROOT}/best.pth" \
        --split "$VIZ_SPLITS" --num "$VIZ_NUM" \
        --out_dir "${OUTPUT_ROOT}/viz" 2>&1 | tee "${LOG_DIR}/viz.log" \
        || echo "[__TAG__] WARN: viz 失败(不阻塞链路, 继续落袋)"
fi

# === step 5: resbag 落袋 ===
python .claude/skills/resbag/resbag.py make \
    --output_root "$OUTPUT_ROOT" --dataset "$DS" \
    --tag "$TAG" --model "$MODEL" \
    --cfg_file "$CFG" --batch_size "$BS_EFF" \
    --note "train_bs=$BS_EFF" 2>&1 | tee "${LOG_DIR}/resbag.log"

[ -f "${OUTPUT_ROOT}/model_store.yaml" ] || { echo "[__TAG__] ERROR: model_store.yaml 未落盘"; exit 1; }

# === step 6: 落 marker (workflow oracle 判 idempotent) ===
touch "$MARKER"
echo "[__TAG__] ALL DONE  $OUTPUT_ROOT  marker=$MARKER  $(date)"
