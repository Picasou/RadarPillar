#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 阶段2 补强：派生 RepDWC 4 档容量 cfg（b5~b8）+ sh 到 experiments/。

与 b1~b4（standard 块、BaseBEVBackbone）成对：OUT_CHANNELS 逐一对齐 b1~b4 的
NUM_FILTERS，backbone 用 RepDWCNoneBackbone（无 neck，首层直出），构成阶段2 容量×
块类型 4×2 对照。

口径说明（写入每个 yaml 顶部注释）：
- RepDWC 深度可分离，同档参数量显著小于 standard；横比成本维度以实测 Params/FLOPs 为准。
- 前端 VFE/Scatter 锁 32（RepDWC input_channels=NUM_BEV_FEATURES 硬约束，见 rep_dwc.py
  audit M4）。b7/b8 首层 OUT_CHANNELS=64 时靠内部 PW 升维 32→64，VFE 不随 standard b3/b4 升 64。
- 这与现有 n3~n6（VFE=64、NUM_BEV=32、首层 OUT=64）已验证配置一致。

复用 derive_rpin_cfgs.py 的 expand/deep_merge/write 策略（展开 _BASE_CONFIG_ 链 →
override → 序列化，不带 _BASE_CONFIG_）。
"""
from pathlib import Path
import copy
import yaml

REPO = Path('.')
BASE_PATH = REPO / 'tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml'
YAML_DIR = REPO / 'experiments' / 'YAML'
SH_DIR = REPO / 'experiments' / 'SH'
YAML_DIR.mkdir(parents=True, exist_ok=True)
SH_DIR.mkdir(parents=True, exist_ok=True)


def expand(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if '_BASE_CONFIG_' in raw:
        base = expand(REPO / raw['_BASE_CONFIG_'])
        del raw['_BASE_CONFIG_']
        return deep_merge(base, raw)
    return raw


def deep_merge(a, b):
    out = {k: v for k, v in a.items()}
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


base_full = expand(BASE_PATH)


def write(tag: str, overrides: dict, note: str = ''):
    merged = deep_merge(base_full, overrides or {})
    merged.pop('_BASE_CONFIG_', None)
    bb = merged.get('MODEL', {}).get('BACKBONE_2D', {})
    # RepDWCNoneBackbone 不消费 BaseBEVBackbone 残留的 NUM_FILTERS/UPSAMPLE_*，
    # 显式 pop 防 cfg 自误导（与 derive_rpin_cfgs.py 的 n4 清理口径一致）。
    if bb.get('NAME', '') == 'RepDWCNoneBackbone':
        for stale in ('NUM_FILTERS', 'UPSAMPLE_STRIDES', 'NUM_UPSAMPLE_FILTERS'):
            bb.pop(stale, None)
    p = YAML_DIR / f'{tag}.yaml'
    with p.open('w', encoding='utf-8') as f:
        if note:
            f.write(note)
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f'  wrote {p.relative_to(REPO)}')


def repdwc_none_cfg(out_channels: list):
    """RepDWCNoneBackbone override：仅 OUT_CHANNELS 变，其余锁默认。

    前端 VFE/NUM_BEV/ATTN 全锁 32（对齐 b1~b4 的最小前端 + RepDWC 输入硬约束）。
    """
    return {
        'MODEL': {
            'VFE': {'NUM_FILTERS': [32]},
            'BACKBONE_3D': {'NAME': 'PillarAttention', 'ATTN_CHANNELS': 32,
                            'NUM_HEADS': 1, 'DROPOUT': 0.0, 'FFN_CHANNELS': 32,
                            'USE_LAYER_NORM': True},
            'MAP_TO_BEV': {'NAME': 'PointPillarScatter', 'NUM_BEV_FEATURES': 32},
            'BACKBONE_2D': {
                'NAME': 'RepDWCNoneBackbone',
                'OUT_CHANNELS': list(out_channels),
                'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
                'NUM_OUTPUTS': 3, 'INFERENCE_MODE': False, 'USE_SE': False,
                'NUM_CONV_BRANCHES': 1, 'USE_NORMCONV': False, 'USE_DWCONV': True,
                # 不设 NUM_FILTERS：RepDWCNoneBackbone 内部走 RepDWCBackbone，只读
                # model_cfg.OUT_CHANNELS，不消费 BaseBEVBackbone 的 NUM_FILTERS。
                # deep_merge 会带入底座残留 NUM_FILTERS，在 write() 里 pop。
            },
        },
    }


# b5~b8：RepDWC 4 档，OUT_CHANNELS 逐一对齐 b1~b4 的 NUM_FILTERS
SPECS = [
    ('b5', [32, 32, 32], 'b1'),
    ('b6', [32, 64, 128], 'b2'),
    ('b7', [64, 128, 256], 'b3'),
    ('b8', [64, 64, 64], 'b4'),
]

NOTE_TMPL = (
    "# RPiN 阶段2 补强：RepDWC 容量扫描（无 neck）。\n"
    "# 对齐 standard 块 {align}（NUM_FILTERS={filters}）；本档 OUT_CHANNELS={filters}。\n"
    "# 口径：RepDWC 深度可分离，同档参数量显著小于 standard；前端 VFE/NUM_BEV 锁 32（RepDWC 输入硬约束），\n"
    "#       首层 OUT=64 时靠 PW 升维 32->64，不随 standard {align} 升 64。横比以 stage_stats 实测 Params/FLOPs 为准。\n"
    "\n"
)

for tag, chans, align in SPECS:
    note = NOTE_TMPL.format(align=align, filters=chans)
    write(tag, repdwc_none_cfg(chans), note=note)

# ---- sh ----
SH_TMPL = '''#!/bin/bash
# RPiN 阶段2 补强完整训练 .sh（80 epochs）— {tag}（RepDWCNoneBackbone，OUT_CHANNELS={filters}，对齐 standard {align}）
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
BS=16
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
'''

for tag, chans, align in SPECS:
    p = SH_DIR / f'train_rpillar_{tag}.sh'
    p.write_text(SH_TMPL.format(tag=tag, filters=chans, align=align), encoding='utf-8')
    p.chmod(0o755)
    print(f'  wrote {p.relative_to(REPO)}')

print(f'\n[derive] b5~b8（4 yaml + 4 sh）已落到 {YAML_DIR} / {SH_DIR}')
