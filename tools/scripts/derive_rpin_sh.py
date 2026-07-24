#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 前置计划 Task 3：派生 22 .sh 到 experiments/SH/。

每 .sh 仿 train_radarpillar.sh 约定：conda activate base + ARGS 数组 + OUTPUT_ROOT + 后台运行。
bs 按 plan §0.5 S11：默认 16；b3/b4/n3/n5/n6/head_center/head_2d 降 8。
.sh 一律在工程根执行（plan §0.5 S16）。
"""
from pathlib import Path

REPO = Path('.')
SH_DIR = REPO / 'experiments' / 'SH'
SH_DIR.mkdir(parents=True, exist_ok=True)

# tag -> (bs, 备注)
SH_MAP = {
    # 阶段1 注意力
    'a1': (16, 'A1=PillarAttention 底座'), 'a0': (16, 'A0=无注意力'),
    'a2': (16, 'A2=SEBlock'), 'a3': (16, 'A3=SEDWConv'),
    # 阶段2 容量
    'b1': (16, 'B1=[32,32,32]'), 'b2': (16, 'B2=[32,64,128]'),
    'b3': (8, 'B3=[64,128,256] 降 bs 8'), 'b4': (8, 'B4=[64,64,64] 降 bs 8'),
    # 阶段3 neck
    'n1': (16, 'N1=standard+无neck'), 'n2': (16, 'N2=standard+FPN(PPFPN)'),
    'n3': (8, 'N3=standard+MDFEN(PPMDFEN) 降 bs 8'),
    'n4': (16, 'N4=RepDWC+无neck(RepDWCNone)'),
    'n5': (8, 'N5=RepDWC+FPN 降 bs 8'),
    'n6': (8, 'N6=RepDWC+MDFEN 降 bs 8'),
    # 阶段4 head
    'head_anchor': (16, 'AnchorHeadSingle'),
    'head_center': (8, 'RadarNeXtCenterHead 降 bs 8'),
    'head_2d': (8, 'RadarNeXtCenterHead2D 降 bs 8'),
    # 阶段5 E/F
    'e1': (16, 'E1 关速度分解'),
    'e2': (16, 'E2 底座(9维)'),
    'e3': (16, 'E3 +VDC'),
    'f1': (16, 'F1 单帧'),
    'f3': (16, 'F3 底座(5frames)'),
}

TEMPLATE = """#!/bin/bash
# RPiN 前置完整训练 .sh（80 epochs）— {tag}（{note}）
# 注：1-epoch 冒烟验证请用 tools/scripts/rpin_1epoch.py（本 .sh 是全量训练入口）。
# 仿 train_radarpillar.sh 约定；conda activate base（plan §0.5 S4：env=base，angle 为死路径）。
set -e
cd "$(dirname "$0")/../.."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$_c/etc/profile.d/conda.sh" ] && {{ source "$_c/etc/profile.d/conda.sh"; break; }}
    done
fi
conda activate base

CFG="experiments/YAML/{tag}.yaml"
BS={bs}
WORKERS=2
EPOCHS=80
GPU=0
EXTRA_TAG="{tag}"
OUTPUT_ROOT="output/train_log/vod/$(date +%Y%m%d%H%M)_radarpillar_{tag}"

ARGS=(
    --cfg_file "$CFG"
    --batch_size "$BS"
    --workers "$WORKERS"
    --epochs "$EPOCHS"
    --extra_tag "$EXTRA_TAG"
    --output_root "$OUTPUT_ROOT"
    --skip_eval
    --set OPTIMIZATION.early_stop.enabled False OPTIMIZATION.LR_WARMUP False
)

LOG="$OUTPUT_ROOT/train_$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUTPUT_ROOT"
echo "[rpin:{tag}] log=$LOG  bs=$BS  output_root=$OUTPUT_ROOT"
nohup python -u tools/train.py "${{ARGS[@]}}" > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
"""


def main():
    for tag, (bs, note) in SH_MAP.items():
        p = SH_DIR / f'train_rpillar_{tag}.sh'
        p.write_text(TEMPLATE.format(tag=tag, bs=bs, note=note), encoding='utf-8')
        p.chmod(0o755)
        print(f'  wrote {p.relative_to(REPO)} (bs={bs})')
    print(f'\n[derive] {len(SH_MAP)} 个 .sh 已落到 {SH_DIR}')


if __name__ == '__main__':
    main()
