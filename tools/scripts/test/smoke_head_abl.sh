#!/usr/bin/env bash
# head ablation ①-⑤ + 真2D 冒烟: mini pkl(160 train/24 val) 1-epoch 训练 + test.py eval 全链路。
# 不触碰原始 pkl; 产物在 .tmp/20260820_head_abl/smoke/<name>/。
# train.py 走 --skip_eval(其 final eval 路径 repeat_eval_ckpt 已被禁用, 生产流程即 test.py 单独 eval)。
# 用法: ./smoke_head_abl.sh [仅跑指定变体名, 如 msr_centerhead3d_noconv]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # 工程根

ABL_DIR=".tmp/20260820_head_abl"
NAMES=(msr_centerhead3d_noconv msr_centerhead3d_k1 msr_centerhead3d_narrow
       msr_centerhead3d_merged msr_centerhead3d_trunk msr_centerhead2d)
if [ -n "${1:-}" ]; then
  NAMES=("${@}")
fi

PASS=(); FAIL=()
for name in "${NAMES[@]}"; do
  out="${ABL_DIR}/smoke/${name}"
  rm -rf "${out}"; mkdir -p "${out}"
  echo "===== [smoke] ${name}: train 1 epoch ====="
  if PYTHONPATH=tools python tools/train.py \
      --cfg_file "${ABL_DIR}/smoke_cfg/${name}_smoke.yaml" \
      --batch_size 4 --workers 2 --epochs 1 --fix_random_seed --skip_eval \
      --output_root "${out}" > "${out}/train_stdout.log" 2>&1; then
    :
  else
    echo "[FAIL] ${name}: train.py 非零退出"
    tail -15 "${out}/train_stdout.log"
    FAIL+=("${name}"); continue
  fi

  log=$(ls "${out}"/log_train_*.txt 2>/dev/null | head -1 || true)
  verdict=""
  [ -z "${log}" ] && verdict="无 log_train 文件; "
  grep -qiE "loss[:=] *(nan|inf)" "${out}/train_stdout.log" "${log}" 2>/dev/null && verdict="${verdict}loss NaN/Inf; "
  grep -q "Traceback" "${out}/train_stdout.log" && verdict="${verdict}train Traceback; "
  ckpt="${out}/ckpt/checkpoint_epoch_1.pth"
  [ -f "${ckpt}" ] || verdict="${verdict}无 epoch1 ckpt; "

  if [ -n "${verdict}" ]; then
    echo "[FAIL] ${name}: ${verdict}"
    tail -5 "${log:-${out}/train_stdout.log}"
    FAIL+=("${name}"); continue
  fi

  echo "----- [smoke] ${name}: test.py eval (predict/decode/NMS/eval 链路) -----"
  if PYTHONPATH=tools python tools/test.py \
      --cfg_file "${ABL_DIR}/smoke_cfg/${name}_smoke.yaml" \
      --ckpt "${ckpt}" \
      --batch_size 4 --workers 2 \
      --output_root "${out}" --extra_tag "eval_smoke" \
      > "${out}/eval_stdout.log" 2>&1; then
    tail -4 "${out}/eval_stdout.log" | sed 's/^/    /'
    echo "[PASS] ${name}"
    PASS+=("${name}")
  else
    echo "[FAIL] ${name}: test.py eval 非零退出"
    tail -15 "${out}/eval_stdout.log"
    FAIL+=("${name}")
  fi
done

echo ""
echo "===== SMOKE SUMMARY: PASS ${#PASS[@]} / $(( ${#PASS[@]} + ${#FAIL[@]} )) ====="
[ ${#PASS[@]} -gt 0 ] && printf '  PASS: %s\n' "${PASS[*]}"
[ ${#FAIL[@]} -gt 0 ] && printf '  FAIL: %s\n' "${FAIL[*]}"
[ ${#FAIL[@]} -eq 0 ]
