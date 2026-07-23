"""RPiN 阶段3 N-2：PPFPN = StandardMultiScale + SecondFPN。
4-bug 修复（对照 radarnext_backbone_fpn.py:105-115 的正确调用）：
  ① SecondFPN.forward 返 [Tensor]，取 [0]
  ② SecondFPN.__init__ 无 input_channels 参，只接 (in_channels, out_channels, upsample_strides, ...)
  ③ multi-scale 输出 large→small（与 RepDWC 顺序一致），SecondFPN 按顺序消费
  ④ num_bev_features = sum(SECOND_FPN.OUT_CHANNELS)
"""
import torch
import torch.nn as nn

from .pp_common import StandardMultiScale
from .second_fpn import SecondFPN


class PPFPNBackbone(nn.Module):
    """Stage 3 N-2：standard 多尺度 + SecondFPN。"""

    def __init__(self, model_cfg, input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        # 子模块分别给 cfg，浅复制避免共享修改
        from easydict import EasyDict
        std_cfg = EasyDict({
            'LAYER_NUMS': list(model_cfg.LAYER_NUMS),
            'LAYER_STRIDES': list(model_cfg.LAYER_STRIDES),
            'NUM_FILTERS': list(model_cfg.NUM_FILTERS),
        })
        self.backbone = StandardMultiScale(std_cfg, input_channels)
        fpn_cfg = model_cfg.SECOND_FPN
        self.fpn = SecondFPN(
            in_channels=list(fpn_cfg.IN_CHANNELS),
            out_channels=list(fpn_cfg.OUT_CHANNELS),
            upsample_strides=list(fpn_cfg.UPSAMPLE_STRIDES),
        )
        # 通道对齐断言（与 RadarNeXtFPNBackbone 一致）
        assert list(model_cfg.NUM_FILTERS) == list(fpn_cfg.IN_CHANNELS), \
            f'NUM_FILTERS({model_cfg.NUM_FILTERS}) 必须 == SECOND_FPN.IN_CHANNELS({fpn_cfg.IN_CHANNELS})'
        self.num_bev_features = int(sum(list(fpn_cfg.OUT_CHANNELS)))

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        multi_scale = self.backbone(x)           # ③ large→small 顺序
        fused = self.fpn(multi_scale)[0]          # ① SecondFPN 返 list，取 [0]
        data_dict['spatial_features_2d'] = fused
        return data_dict
