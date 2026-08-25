#!/usr/bin/env bash
# 泛化集(model_1)全量评测入口:
#   1) 每模型渲染 tracker/cfg/cfg_gen_<tag>.yaml (MODEL 指向 resbag, DATA 指向该模型专属数据拷贝)
#   2) do_tracker mode=1/save=1/overlap=1 → 回灌检测 bin (0200/0201.00000.bin) 写入拷贝目录
#   3) eval_generalization.py → 总体+分序列 mAP 报告 (读 model_1 原始数据, 不受回灌覆盖影响)
# 用法: ./tools/scripts/test/run_generalization.sh [tag ...]  (默认全部 6 模型)
set -euo pipefail
cd "$(dirname "$0")/../../../"   # 工程根

GEN_ROOT=/mnt/d/DataSet/.generalization
SRC_DATA=$GEN_ROOT/model_1
REPORT=output/generalization_eval_report.md

declare -A RUNS=(
  [msr_radarpillar]=202608181912_rpillar_msr_msr
  [msr_radarNeXt]=202608221828_msr_radarNeXt_msr_radarNeXt
  [msr_centerhead2d_trunk]=202608220357_rpillar_msr_centerhead2d_trunk_msr_centerhead2d_trunk
  [msr_pp64_ch2d_trunk]=202608220918_rpillar_msr_pp64_ch2d_trunk_msr_pp64_ch2d_trunk
  [msr_ppmix_ch2d_trunk]=202608221247_rpillar_msr_ppmix_ch2d_trunk_msr_ppmix_ch2d_trunk
  [msr_pp333mix_ch2d_trunk]=202608231928_rpillar_msr_pp333mix_ch2d_trunk_msr_pp333mix_ch2d_trunk
)

TAGS=("$@")
[ ${#TAGS[@]} -eq 0 ] && TAGS=(msr_radarpillar msr_radarNeXt msr_centerhead2d_trunk msr_pp64_ch2d_trunk msr_ppmix_ch2d_trunk msr_pp333mix_ch2d_trunk)

# ---- 1. 数据拷贝(已存在则跳过) ----
for tag in "${TAGS[@]}"; do
  if [ ! -d "$GEN_ROOT/$tag" ]; then
    echo "[copy] model_1 -> $tag"
    cp -r "$SRC_DATA" "$GEN_ROOT/$tag"
  fi
done

# ---- 2. 回灌: 每模型渲染 cfg + do_tracker (标记文件存在则跳过, 重跑删 $GEN_ROOT/<tag>/.gen_done) ----
for tag in "${TAGS[@]}"; do
  if [ -f "$GEN_ROOT/$tag/.gen_done" ]; then
    echo "[tracker] $tag 已回灌, 跳过"
    continue
  fi
  run=${RUNS[$tag]}
  resbag=output/train_log/msr/$run/resbag
  cfg_out=tracker/cfg/cfg_gen_$tag.yaml
  python3 - "$tag" "$resbag" "$GEN_ROOT/$tag" "$cfg_out" << 'PY'
import sys
tag, resbag, data_dir, cfg_out = sys.argv[1:5]
lines = open('tracker/cfg/cfg.yaml').read().splitlines()
out, in_paths = [], False
for ln in lines:
    if ln.strip() == 'paths:':
        out.append('  paths:')
        for p in sorted(__import__('os').listdir(data_dir)):
            out.append('    - %s/%s' % (data_dir, p))
        in_paths = True
        continue
    if in_paths and ln.strip().startswith('- '):
        continue
    if in_paths and ln and not ln.startswith(' '):
        in_paths = False
    if ln.strip().startswith('cfg:'):
        out.append('  cfg: /home/admin/projects/RadarPillar/%s/cfg.yaml' % resbag)
    elif ln.strip().startswith('ckpt:'):
        out.append('  ckpt: /home/admin/projects/RadarPillar/%s/best.pth' % resbag)
    else:
        out.append(ln)
open(cfg_out, 'w').write('\n'.join(out) + '\n')
print('[cfg] %s -> %s' % (tag, cfg_out))
PY
  echo "[tracker] $tag 回灌落盘 ..."
  PYTHONPATH=tools python tools/do_tracker.py --cfg "$cfg_out" 2>&1 | tail -3
  touch "$GEN_ROOT/$tag/.gen_done"
done

# ---- 3. mAP 评测 (每模型独立进程: pcdet 全局 cfg 就地合并, 同进程连建多模型会交叉污染) ----
rm -f "$REPORT"
for tag in "${TAGS[@]}"; do
  echo "[eval] $tag mAP ..."
  PYTHONPATH=tools python tools/scripts/test/eval_generalization.py \
    --data_root "$SRC_DATA" --models "$tag" --report "$REPORT" --append
done

echo "=== done, report: $REPORT ==="
