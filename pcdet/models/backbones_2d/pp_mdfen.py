"""RPiN 阶段3 N-3：PPMDFEN = StandardMultiScale + MDFENNeck。
4-bug 修复同上；额外约束：CHANNELS_LIST[0:3] 必须 == NUM_FILTERS（plan 引用 radarnext_backbone_mdfen.py:125）。
channels_list[3:7] 是 PAN 内部通道，照搬 RadarNeXt 原值 [64,128,256,128,64,128,256]。
"""
import torch
import torch.nn as nn
from easydict import EasyDict

from .pp_common import StandardMultiScale
from .mdfen_neck import MDFENNeck


class PPMDFENBackbone(nn.Module):
    """Stage 3 N-3：standard 多尺度 + MDFENNeck。"""

    def __init__(self, model_cfg, input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        std_cfg = EasyDict({
            'LAYER_NUMS': list(model_cfg.LAYER_NUMS),
            'LAYER_STRIDES': list(model_cfg.LAYER_STRIDES),
            'NUM_FILTERS': list(model_cfg.NUM_FILTERS),
        })
        self.backbone = StandardMultiScale(std_cfg, input_channels)
        neck_cfg = model_cfg.MDFEN_NECK
        # 通道对齐断言
        assert list(model_cfg.NUM_FILTERS) == list(neck_cfg.CHANNELS_LIST[0:3]), \
            f'NUM_FILTERS({model_cfg.NUM_FILTERS}) 必须 == MDFEN_NECK.CHANNELS_LIST[0:3]({list(neck_cfg.CHANNELS_LIST[0:3])})'
        self.neck = MDFENNeck(
            channels_list=list(neck_cfg.CHANNELS_LIST),
            num_repeats=list(neck_cfg.NUM_REPEATS),
            dcn_layer=bool(neck_cfg.get('DCN_LAYER', False)),
            former=bool(neck_cfg.get('FORMER', True)),
            latter=bool(neck_cfg.get('LATTER', False)),
            group=int(neck_cfg.get('GROUP', 4)),
            use_ffn=bool(neck_cfg.get('USE_FFN', False)),
            use_norm=bool(neck_cfg.get('USE_NORM', False)),
            inference_mode=bool(neck_cfg.get('INFERENCE_MODE', False)),
            use_se=bool(neck_cfg.get('USE_SE', False)),
            num_conv_branches=int(neck_cfg.get('NUM_CONV_BRANCHES', 1)),
            use_dwconv=bool(neck_cfg.get('USE_DWCONV', True)),
            use_normconv=bool(neck_cfg.get('USE_NORMCONV', False)),
            multi_fusion=bool(neck_cfg.get('MULTI_FUSION', True)),
            fused_channels=list(neck_cfg.get('FUSED_CHANNELS', [128, 128, 128])),
            fusion_strides=list(neck_cfg.get('FUSION_STRIDES', [1, 2])),
        )
        # num_bev_features 与 radarnext_backbone_mdfen 一致：multi_fusion→sum(fused_channels)，否则 channels_list[6]
        self.num_bev_features = (int(sum(neck_cfg.FUSED_CHANNELS))
                                 if neck_cfg.get('MULTI_FUSION', True)
                                 else int(neck_cfg.CHANNELS_LIST[6]))

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        multi_scale = self.backbone(x)          # large→small
        fused = self.neck(multi_scale)[0]       # ① MDFENNeck 返 list，取 [0]
        data_dict['spatial_features_2d'] = fused
        return data_dict
