"""RadarNeXt FPN backbone — wraps RepDWC + SECONDFPN as ONE BACKBONE_2D.

OpenPCDet's ``module_topology`` has no independent "neck" slot, so the
RadarNeXt FPN-variant model (RepDWC backbone + SECONDFPN neck in mmdet3d) is
folded into a single ``BACKBONE_2D`` module here.

Pipeline:

    data_dict['spatial_features']  (B, 32, 320, 320)
        -> RepDWCBackbone  -> 3 scales [(B,64,160,160),(B,128,80,80),(B,256,40,40)]
        -> SecondFPN       -> fused    (B, 384, 80, 80)
    data_dict['spatial_features_2d'] = fused

The wrapped ``RepDWCBackbone`` and ``SecondFPN`` are constructed from
sub-dicts of ``model_cfg`` so the YAML config mirrors the original mmdet3d
``backbone`` / ``neck`` blocks:

    BACKBONE_2D:
        NAME: RadarNeXtFPNBackbone
        REP_DWC:               # <- original mmdet3d 'backbone' block
            ...
        SECOND_FPN:            # <- original mmdet3d 'neck' block
            IN_CHANNELS: [64, 128, 256]
            OUT_CHANNELS: [128, 128, 128]
            UPSAMPLE_STRIDES: [0.5, 1, 2]
            USE_CONV_FOR_NO_STRIDE: True
            NORM_CFG: {EPS: 0.001, MOMENTUM: 0.01}
            UPSAMPLE_CFG: {BIAS: False}
            CONV_CFG: {BIAS: False}

``num_bev_features`` is exposed (mirroring ``BaseBEVBackbone``) so OpenPCDet's
``build_backbone_2d`` can wire up downstream modules; it is set to the FPN's
output channel count (``sum(OUT_CHANNELS)`` = 384) since that is the channel
count of the 2D spatial feature map this module emits.
"""

import torch
from torch import nn as nn

from .rep_dwc import RepDWCBackbone
from .second_fpn import SecondFPN


class RadarNeXtFPNBackbone(nn.Module):
    """RepDWC + SECONDFPN fused into a single BACKBONE_2D module.

    Args:
        model_cfg (EasyDict): YAML ``BACKBONE_2D`` block. Must contain
            sub-dicts ``REP_DWC`` and ``SECOND_FPN`` (see module docstring).
        input_channels (int): Number of BEV feature channels fed in by the
            vfe→middle_encoder stack. Audit decision M4: this is
            ``NUM_BEV_FEATURES = 32`` (not the 64 in the original RadarNeXt
            mmdet3d config).

    Forward contract (matches OpenPCDet's ``BACKBONE_2D`` convention):
        Input:  ``data_dict`` carrying ``spatial_features`` ``(N, C_in, H, W)``.
        Output: the same ``data_dict`` with ``spatial_features_2d`` set to the
                fused ``(N, sum(OUT_CHANNELS), H/4, W/4)`` feature.
    """

    def __init__(self, model_cfg, input_channels: int = 32):
        super(RadarNeXtFPNBackbone, self).__init__()
        self.model_cfg = model_cfg

        # --- RepDWC backbone (produces 3 BEV scales) ---
        # RepDWCBackbone reads its own hyperparameters from model_cfg.REP_DWC.
        rep_dwc_cfg = model_cfg.REP_DWC
        self.backbone = RepDWCBackbone(
            model_cfg=rep_dwc_cfg,
            input_channels=input_channels,
        )

        # --- SECONDFPN neck (fuses 3 scales -> 1) ---
        fpn_cfg = model_cfg.SECOND_FPN
        in_channels = list(fpn_cfg.IN_CHANNELS)
        out_channels = list(fpn_cfg.OUT_CHANNELS)
        upsample_strides = list(fpn_cfg.UPSAMPLE_STRIDES)
        use_conv_for_no_stride = fpn_cfg.get('USE_CONV_FOR_NO_STRIDE', False)
        # mmdet3d defaults preserved if absent.
        norm_cfg = dict(fpn_cfg.get('NORM_CFG', dict(type='BN', eps=1e-3, momentum=0.01)))
        norm_cfg['eps'] = norm_cfg.get('eps', 1e-3)
        norm_cfg['momentum'] = norm_cfg.get('momentum', 0.01)
        upsample_cfg = dict(fpn_cfg.get('UPSAMPLE_CFG', dict(type='deconv', bias=False)))
        conv_cfg = dict(fpn_cfg.get('CONV_CFG', dict(type='Conv2d', bias=False)))

        self.fpn = SecondFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_strides=upsample_strides,
            norm_cfg=norm_cfg,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            use_conv_for_no_stride=use_conv_for_no_stride,
        )

        # Exposed for OpenPCDet detector assembly (downstream dense heads read
        # this to size their input projection). It is the FPN's fused channel
        # count: sum(out_channels) = 3 * 128 = 384 for the FPN-variant config.
        self.num_bev_features = int(sum(out_channels))
        # Sanity: in/out channel lists must align across RepDWC tails and FPN heads.
        assert list(rep_dwc_cfg.OUT_CHANNELS) == in_channels, \
            'REP_DWC.OUT_CHANNELS must equal SECOND_FPN.IN_CHANNELS ' \
            f'(got {list(rep_dwc_cfg.OUT_CHANNELS)} vs {in_channels})'

    def forward(self, data_dict):
        """Run RepDWC -> SecondFPN and write ``spatial_features_2d``."""
        spatial_features = data_dict['spatial_features']  # (N, 32, 320, 320)

        # RepDWC: produces list of 3 multi-scale features.
        multi_scale_feats = self.backbone(spatial_features)
        # SecondFPN: fuses them into a single (N, 384, 80, 80) tensor.
        fused = self.fpn(multi_scale_feats)[0]

        data_dict['spatial_features_2d'] = fused
        return data_dict
