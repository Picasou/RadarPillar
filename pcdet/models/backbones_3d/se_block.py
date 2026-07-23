"""RPiN 阶段1 A2 候选：SE bottleneck 通道重标定。
build_se_bottleneck 为模块级可复用工厂：A3 (SEDWConv) 复用之。
契约（BACKBONE_3D 槽）：`__init__(model_cfg, input_channels, **kwargs)`，
forward 读 `pillar_features`，就地重标定后写回。设 `self.num_point_features`。
"""
import torch
import torch.nn as nn


def build_se_bottleneck(channels: int, reduction: int = 4) -> nn.Sequential:
    """SE 瓶颈：FC(C→C/r)→ReLU→FC(C/r→C)→Sigmoid。
    A2 直接用此模块做通道重标定；A3 在 DWConv 之后用同一模块做门控。
    """
    hidden = max(channels // reduction, 1)
    return nn.Sequential(
        nn.Linear(channels, hidden, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, channels, bias=False),
        nn.Sigmoid(),
    )


class SEBlock(nn.Module):
    """A2：纯通道注意力，BEV-GAP ≡ pillar 集均值（差常数因子，被 FC 权重吸收）。"""

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        attn_channels = int(model_cfg.get('ATTN_CHANNELS', input_channels))
        reduction = int(model_cfg.get('REDUCTION', 4))
        self.num_point_features = attn_channels
        # 输入与注意力通道不同时做线性对齐（与 PillarAttention 一致做法）
        self.pre_mlp = (nn.Linear(input_channels, attn_channels, bias=False)
                        if input_channels != attn_channels else nn.Identity())
        self.bottleneck = build_se_bottleneck(attn_channels, reduction)

    def forward(self, batch_dict):
        pf = batch_dict['pillar_features']        # (M, C_in)
        h = self.pre_mlp(pf)                      # (M, C_attn)
        w = self.bottleneck(h.mean(dim=0, keepdim=True))   # (1, C_attn)
        batch_dict['pillar_features'] = h * w
        return batch_dict
