#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RPiN 前置计划 Task 3：派生 22 cfg 到 experiments/YAML/。

策略：递归展开底座的 `_BASE_CONFIG_` 链 → 得到一份完全可解析的 cfg dict，
然后对该 dict 应用每 cfg 的覆写，序列化落盘（**不带** `_BASE_CONFIG_`）。
避免双层链 merge 不到的问题，也避免每个 cfg 重复内联大块底座。
"""
from pathlib import Path
import copy
import yaml

REPO = Path('.')
BASE_PATH = REPO / 'tools/cfgs/model/vod_models/radarpillar/vod_radarpillar.yaml'
YAML_DIR = REPO / 'experiments' / 'YAML'
YAML_DIR.mkdir(parents=True, exist_ok=True)


def expand(path: Path) -> dict:
    """递归把 yaml 的 _BASE_CONFIG_ 链展开成单层 dict。"""
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


def anchor_cfg_with_stride(stride: int):
    """复用底座 ANCHOR_GENERATOR_CONFIG，仅改 feature_map_stride。

    MDFEN（n3/n6）multi_fusion 融合到 80x80（grid/4），anchor 网格须同步到
    stride=4 才与 backbone 输出对齐（其余 neck 输出 160x160，用底座 stride=2）。
    """
    anchors = copy.deepcopy(base_full['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG'])
    for a in anchors:
        a['feature_map_stride'] = stride
    return anchors


def write(tag: str, overrides: dict):
    merged = deep_merge(base_full, overrides or {})
    # 移除顶层 _BASE_CONFIG_（已展开）
    merged.pop('_BASE_CONFIG_', None)
    # 卫生：RepDWC 系列 backbone（RadarNeXt*/RepDWCNone）用 REP_DWC.OUT_CHANNELS +
    # neck 子配置，不消费 BaseBEVBackbone 的 NUM_FILTERS/UPSAMPLE_*。这些键是 baseline
    # deep_merge 带入的残留（[32,32,32] 与实际 64 通道自相矛盾），显式 pop 防 cfg 自误导。
    bb = merged.get('MODEL', {}).get('BACKBONE_2D', {})
    # 仅 RepDWC 系列（RadarNeXt*/RepDWCNone）用 REP_DWC.OUT_CHANNELS，不消费
    # UPSAMPLE_*（BaseBEVBackbone 残留）。PP* 仍需 NUM_FILTERS（StandardMultiScale）。
    if bb.get('NAME', '') in ('RadarNeXtFPNBackbone', 'RadarNeXtMDFENBackbone', 'RepDWCNoneBackbone'):
        for stale in ('UPSAMPLE_STRIDES', 'NUM_UPSAMPLE_FILTERS'):
            bb.pop(stale, None)
    p = YAML_DIR / f'{tag}.yaml'
    with p.open('w', encoding='utf-8') as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f'  wrote {p.relative_to(REPO)}')


# Stage 1 注意力
P1 = {
    'MODEL': {
        'BACKBONE_3D': {
            'NAME': 'PillarAttention', 'ATTN_CHANNELS': 32, 'NUM_HEADS': 1,
            'DROPOUT': 0.0, 'FFN_CHANNELS': 32, 'USE_LAYER_NORM': True,
        },
        'BACKBONE_2D': {
            'NAME': 'BaseBEVBackbone',
            'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
            'NUM_FILTERS': [32, 32, 32], 'UPSAMPLE_STRIDES': [1, 2, 4],
            'NUM_UPSAMPLE_FILTERS': [32, 32, 32],
        },
    },
}
write('a1', P1)
write('a0', {'MODEL': {'BACKBONE_3D': None}})
write('a2', {'MODEL': {'BACKBONE_3D': {'NAME': 'SEBlock', 'ATTN_CHANNELS': 32, 'REDUCTION': 4}}})
write('a3', {'MODEL': {'BACKBONE_3D': {'NAME': 'SEDWConv', 'ATTN_CHANNELS': 32, 'REDUCTION': 4, 'DW_KERNEL': 3}}})

# Stage 2 容量
def capacity_cfg(backbone_filters, vfe_filters):
    return {
        'MODEL': {
            'VFE': {'NUM_FILTERS': vfe_filters},
            'MAP_TO_BEV': {'NUM_BEV_FEATURES': backbone_filters[0]},
            'BACKBONE_2D': {'NUM_FILTERS': backbone_filters,
                            'NUM_UPSAMPLE_FILTERS': list(backbone_filters)},
            'BACKBONE_3D': {'ATTN_CHANNELS': backbone_filters[0],
                             'FFN_CHANNELS': backbone_filters[0]},
        },
    }


write('b1', P1)
write('b2', capacity_cfg([32, 64, 128], [32]))
write('b3', capacity_cfg([64, 128, 256], [64]))
write('b4', capacity_cfg([64, 64, 64], [64]))

# Stage 3 neck
write('n1', P1)
write('n2', {
    'MODEL': {
        'BACKBONE_2D': {
            'NAME': 'PPFPNBackbone',
            'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
            'NUM_FILTERS': [32, 64, 128],
            'SECOND_FPN': {
                'IN_CHANNELS': [32, 64, 128], 'OUT_CHANNELS': [128, 128, 128],
                'UPSAMPLE_STRIDES': [1, 2, 4],
            },
        },
    },
})
write('n3', {
    'MODEL': {
        'BACKBONE_3D': {'ATTN_CHANNELS': 64, 'FFN_CHANNELS': 64},
        'VFE': {'NUM_FILTERS': [64]},
        'MAP_TO_BEV': {'NUM_BEV_FEATURES': 64},
        'BACKBONE_2D': {
            'NAME': 'PPMDFENBackbone',
            'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
            'NUM_FILTERS': [64, 128, 256],
            'NUM_UPSAMPLE_FILTERS': [64, 128, 256],
            'MDFEN_NECK': {
                'CHANNELS_LIST': [64, 128, 256, 128, 64, 128, 256],
                'NUM_REPEATS': [1, 1, 1, 1],
                'DCN_LAYER': False, 'FORMER': True, 'LATTER': False, 'GROUP': 4,
                'MULTI_FUSION': True, 'FUSED_CHANNELS': [128, 128, 128], 'FUSION_STRIDES': [1, 2],
                'USE_DWCONV': True, 'USE_NORMCONV': False,
            },
        },
        'DENSE_HEAD': {'ANCHOR_GENERATOR_CONFIG': anchor_cfg_with_stride(4)},
    },
})
write('n4', {
    'MODEL': {
        'BACKBONE_3D': {'ATTN_CHANNELS': 64, 'FFN_CHANNELS': 64},
        'VFE': {'NUM_FILTERS': [64]},
        'MAP_TO_BEV': {'NUM_BEV_FEATURES': 64},
        'BACKBONE_2D': {
            'NAME': 'RepDWCNoneBackbone',
            'OUT_CHANNELS': [64, 128, 256],
            'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
            'NUM_OUTPUTS': 3, 'INFERENCE_MODE': False, 'USE_SE': False,
            'NUM_CONV_BRANCHES': 1, 'USE_NORMCONV': False, 'USE_DWCONV': True,
            'NUM_UPSAMPLE_FILTERS': [64, 128, 256],
        },
        # n4 取 outs[0]=160×160（见 repdwc_none.py 设计裁决），anchor stride 与 n1 同=2
    },
})
write('n5', {
    'MODEL': {
        'BACKBONE_3D': {'ATTN_CHANNELS': 64, 'FFN_CHANNELS': 64},
        'VFE': {'NUM_FILTERS': [64]},
        'MAP_TO_BEV': {'NUM_BEV_FEATURES': 64},
        'BACKBONE_2D': {
            'NAME': 'RadarNeXtFPNBackbone',
            'REP_DWC': {
                'OUT_CHANNELS': [64, 128, 256], 'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
                'NUM_OUTPUTS': 3, 'INFERENCE_MODE': False, 'USE_SE': False,
                'NUM_CONV_BRANCHES': 1, 'USE_NORMCONV': False, 'USE_DWCONV': True,
            },
            'SECOND_FPN': {
                'IN_CHANNELS': [64, 128, 256], 'OUT_CHANNELS': [128, 128, 128],
                'UPSAMPLE_STRIDES': [1, 2, 4],
            },
        },
    },
})
write('n6', {
    'MODEL': {
        'BACKBONE_3D': {'ATTN_CHANNELS': 64, 'FFN_CHANNELS': 64},
        'VFE': {'NUM_FILTERS': [64]},
        'MAP_TO_BEV': {'NUM_BEV_FEATURES': 64},
        'BACKBONE_2D': {
            'NAME': 'RadarNeXtMDFENBackbone',
            'REP_DWC': {
                'OUT_CHANNELS': [64, 128, 256], 'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2],
                'NUM_OUTPUTS': 3, 'INFERENCE_MODE': False, 'USE_SE': False,
                'NUM_CONV_BRANCHES': 1, 'USE_NORMCONV': False, 'USE_DWCONV': True,
            },
            'MDFEN_NECK': {
                'CHANNELS_LIST': [64, 128, 256, 128, 64, 128, 256],
                'NUM_REPEATS': [1, 1, 1, 1],
                'DCN_LAYER': False, 'FORMER': True, 'LATTER': False, 'GROUP': 4,
                'MULTI_FUSION': True, 'FUSED_CHANNELS': [128, 128, 128], 'FUSION_STRIDES': [1, 2],
                'USE_DWCONV': True, 'USE_NORMCONV': False,
            },
        },
        'DENSE_HEAD': {'ANCHOR_GENERATOR_CONFIG': anchor_cfg_with_stride(4)},
    },
})

# Stage 4 head
write('head_anchor', P1)
write('head_center', {
    'MODEL': {
        'DENSE_HEAD': {
            'NAME': 'RadarNeXtCenterHead',
            'TASKS': [{'num_class': 3, 'class_names': ['Car', 'Pedestrian', 'Cyclist']}],
            'CODE_WEIGHTS': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'COMMON_HEADS': {'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2)},
            'OUT_SIZE_FACTOR': 2,
            # H1 修复：RadarNeXtCenterHead.post_processing 读 model_cfg.NMS_CONFIG（无默认），
            # 缺之则 eval/inference 必崩（AttributeError）。镜像 head_2d / vod_radarnext_fpn。
            'NMS_CONFIG': {'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.1,
                            'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500},
            'SCORE_THRESHOLD': 0.2,
            'POST_CENTER_LIMIT_RANGE': [0, -25.6, -3, 51.2, 25.6, 2],
        },
    },
})
write('head_2d', {
    'MODEL': {
        'DENSE_HEAD': {
            'NAME': 'RadarNeXtCenterHead2D',
            'TASKS': [{'num_class': 3, 'class_names': ['Car', 'Pedestrian', 'Cyclist']}],
            'CODE_WEIGHTS': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'COMMON_HEADS': {'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'vel': (2, 2), 'rot': (2, 2)},
            'WEIGHT': 1.0, 'IOU_WEIGHT': 1.0, 'IOU_REG_WEIGHT': 0.5,
            # RPiN 修复：head_2D 跑在 BaseBEVBackbone 输出之上（spatial=160x160）；
            # OUT_SIZE_FACTOR=2 → feature_map=grid/2=160x160 与 pred 匹配（同 head_center）。
            # 旧版误设 OUT_SIZE_FACTOR=1（以为 backbone=320x320）→ feature_map=320 与 160 pred 2x 失配。
            'OUT_SIZE_FACTOR': 2,
            'ANCHOR_BOTTOM_HEIGHTS': [-1.78, -0.6, -0.72],
            'NMS_CONFIG': {'NMS_TYPE': 'nms_gpu', 'NMS_THRESH': 0.1,
                            'NMS_PRE_MAXSIZE': 4096, 'NMS_POST_MAXSIZE': 500},
            'SCORE_THRESHOLD': 0.2,
            'POST_CENTER_LIMIT_RANGE': [0, -25.6, -3, 51.2, 25.6, 2],
        },
    },
})

# Stage 5 E/F
write('e2', P1)
write('e1', {
    'MODEL': {
        'VFE': {
            'USE_VELOCITY_DECOMPOSITION': False,
            'USE_REL_VELOCITY_DECOMPOSITION': False,
            'USE_VELOCITY_OFFSET': False,
            'USE_REL_VELOCITY_OFFSET': False,
        },
    },
})
write('e3', {'DATA_CONFIG': {'USE_VDC': True,
                              # H4 修复：radar_5frames 的 time 列是整数帧索引 [-4..0]（非秒）；
                              # 10Hz 同步网格 → 帧间隔 0.1s，time_scale=0.1 把帧索引换算成秒。
                              # 默认 1.0 会把帧索引当秒 → 运动补偿过量 ~10×、7.5% 点甩出 PC range。
                              'VDC_CFG': {'time_scale': 0.1, 'use_vr_comp': True}}})
write('f1', {
    'DATA_CONFIG': {
        'DATA_PATH': './data/VoD/view_of_delft_PUBLIC/radar_1frame',
        'INFO_PATH': {'train': ['vod_infos_train.pkl'], 'test': ['vod_infos_val.pkl']},
    },
})
write('f3', P1)

print(f'\n[derive] 全 22 cfg 已落到 {YAML_DIR}')
