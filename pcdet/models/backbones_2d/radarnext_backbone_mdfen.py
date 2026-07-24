"""RadarNeXt MDFEN backbone — wraps RepDWC + MDFENNeck as ONE BACKBONE_2D.

OpenPCDet's ``module_topology`` has no independent "neck" slot, so the
RadarNeXt MDFEN-variant model (RepDWC backbone + MDFENNeck neck in mmdet3d)
is folded into a single ``BACKBONE_2D`` module here — mirroring
``radarnext_backbone_fpn.py`` (Task 3).

Pipeline:

    data_dict['spatial_features']            (B, 32, 320, 320)
        -> RepDWCBackbone -> 3 scales
           [(B, 64, 160, 160), (B, 128, 80, 80), (B, 256, 40, 40)]
           (mmdet3d ordering: largest-first (x0,x1,x2) = (160,80,40))
        -> MDFENNeck      -> fused           (B, 384, 80, 80)
           (PAN bidirectional + former_deform2 DCN + MultiMAPFusion
            fusion_strides=[1,2] -> 全部融合到中间尺度 80×80=grid/4)
           注：head 对齐口径——anchor cfg feature_map_stride=4（或 center head
           SepHead STRIDES=[2] 上采样 80→160）。早期 docstring 误标 160×160 已更正。
    data_dict['spatial_features_2d'] = fused

The wrapped ``RepDWCBackbone`` and ``MDFENNeck`` are constructed from
sub-dicts of ``model_cfg`` so the YAML config mirrors the original mmdet3d
``backbone`` / ``neck`` blocks:

    BACKBONE_2D:
        NAME: RadarNeXtMDFENBackbone
        REP_DWC:               # <- original mmdet3d 'backbone' block
            ...
        MDFEN_NECK:            # <- original mmdet3d 'neck' block (MDFENNeck)
            CHANNELS_LIST:   [64, 128, 256, 128, 64, 128, 256]
            NUM_REPEATS:     [1, 1, 1, 1]
            DCN_LAYER:       False
            DCN_INDEX:       [1]
            DCN_IDS:         [2]
            FORMER:          True
            LATTER:          False
            GROUP:           4
            USE_FFN:         False
            USE_NORM:        False
            INFERENCE_MODE:  False
            USE_SE:          False
            NUM_CONV_BRANCHES: 1
            USE_DWCONV:      True
            USE_NORMCONV:    False
            MULTI_FUSION:    True
            FUSED_CHANNELS:  [128, 128, 128]
            FUSION_STRIDES:  [1, 2]
"""

import torch
from torch import nn as nn

from .rep_dwc import RepDWCBackbone
from .mdfen_neck import MDFENNeck


class RadarNeXtMDFENBackbone(nn.Module):
    """RepDWC + MDFENNeck fused into a single BACKBONE_2D module.

    Forward contract (matches OpenPCDet's ``BACKBONE_2D`` convention):
        Input:  ``data_dict`` carrying ``spatial_features`` ``(N, C_in, H, W)``.
        Output: the same ``data_dict`` with ``spatial_features_2d`` set to the
                fused ``(N, sum(FUSED_CHANNELS), 160, 160)`` feature.
    """

    def __init__(self, model_cfg, input_channels: int = 32):
        super(RadarNeXtMDFENBackbone, self).__init__()
        self.model_cfg = model_cfg

        # --- RepDWC backbone (produces 3 BEV scales) ---
        rep_dwc_cfg = model_cfg.REP_DWC
        self.backbone = RepDWCBackbone(
            model_cfg=rep_dwc_cfg,
            input_channels=input_channels,
        )

        # --- MDFENNeck (PAN bidirectional + DCN + MultiMAPFusion) ---
        ncfg = model_cfg.MDFEN_NECK
        channels_list = list(ncfg.CHANNELS_LIST)
        num_repeats = list(ncfg.NUM_REPEATS)
        dcn_layer = bool(ncfg.get('DCN_LAYER', True))
        dcn_index = list(ncfg.get('DCN_INDEX', [0]))
        dcn_ids = list(ncfg.get('DCN_IDS', [2]))
        former = bool(ncfg.get('FORMER', True))
        latter = bool(ncfg.get('LATTER', False))
        group = int(ncfg.get('GROUP', 4))
        use_ffn = bool(ncfg.get('USE_FFN', False))
        use_norm = bool(ncfg.get('USE_NORM', False))
        inference_mode = bool(ncfg.get('INFERENCE_MODE', False))
        use_se = bool(ncfg.get('USE_SE', False))
        num_conv_branches = int(ncfg.get('NUM_CONV_BRANCHES', 1))
        use_dwconv = bool(ncfg.get('USE_DWCONV', True))
        use_normconv = bool(ncfg.get('USE_NORMCONV', False))
        multi_fusion = bool(ncfg.get('MULTI_FUSION', True))
        fused_channels = list(ncfg.get('FUSED_CHANNELS', [128, 128, 128]))
        fusion_strides = list(ncfg.get('FUSION_STRIDES', [1, 2]))

        self.neck = MDFENNeck(
            channels_list=channels_list,
            num_repeats=num_repeats,
            dcn_layer=dcn_layer,
            dcn_index=dcn_index,
            dcn_ids=dcn_ids,
            former=former,
            latter=latter,
            group=group,
            use_ffn=use_ffn,
            use_norm=use_norm,
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv,
            multi_fusion=multi_fusion,
            fused_channels=fused_channels,
            fusion_strides=fusion_strides,
        )

        # Exposed for detector assembly: the MDFEN fused channel count
        # (sum(FUSED_CHANNELS) = 384 for the production config). Downstream
        # dense heads read this to size their input projection.
        self.num_bev_features = int(sum(fused_channels)) if multi_fusion \
            else channels_list[6]
        # Sanity: RepDWC tails (x0, x1, x2 from largest to smallest) must
        # align with channels_list[2], [1], [0] respectively (the PAN
        # top-down consumes x2=smallest first).
        assert list(rep_dwc_cfg.OUT_CHANNELS) == \
            [channels_list[0], channels_list[1], channels_list[2]], \
            'REP_DWC.OUT_CHANNELS must equal MDFEN_NECK.CHANNELS_LIST[0:3] ' \
            f'(got {list(rep_dwc_cfg.OUT_CHANNELS)} vs ' \
            f'{[channels_list[0], channels_list[1], channels_list[2]]})'

    def forward(self, data_dict):
        """Run RepDWC -> MDFENNeck and write ``spatial_features_2d``."""
        spatial_features = data_dict['spatial_features']  # (N, 32, 320, 320)

        # RepDWC: produces [block0_out, block1_out, block2_out] = the 3 BEV
        # scales in FORWARD (largest-spatial-first) order:
        #   [(B, 64, 160, 160), (B, 128, 80, 80), (B, 256, 40, 40)]
        # The mmdet3d MDFENNeck forward consumes them in the order
        #   (x2, x1, x0) = (largest-spatial/least-channels,
        #                   mid,
        #                   smallest-spatial/most-channels),
        # which is EXACTLY RepDWC's forward order (x2=64ch@160, x1=128ch@80,
        # x0=256ch@40). So NO reorder is needed.
        multi_scale_feats = self.backbone(spatial_features)

        fused = self.neck(multi_scale_feats)[0]

        data_dict['spatial_features_2d'] = fused
        return data_dict
