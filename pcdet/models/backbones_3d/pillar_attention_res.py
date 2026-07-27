"""PillarAttention-Res：PillarAttention 的 ResNet 残差变体（RPiN 阶段1 a4）。

动机：原 PillarAttention（a1）整体覆盖 pillar_features，PillarVFE 输出的干净
几何特征（xyz/RCS/速度分解）被 attention 重新混合后变模糊，导致 Car AP 反而
低于无注意力基线 a0（36.18 vs 37.69）。a4 改为 ResNet 式短接：

    out = short_cut(B1) + TransformerBlock(B1)

短接保留原始几何支路（恒等映射优先），attention 只提供增量修正。最坏情况下
attention 退化为零，a4 退化为 a0（纯透传）；理想情况下叠加语义增益。

通道对齐：当 ATTN_CHANNELS == 输入通道时，short_cut = Identity（零成本、零参数）；
不一致时 short_cut = 1x1 Linear 对齐通道（标准 ResNet short-cut 做法）。
TransformerBlock（attn+FFN+norm）完全复用父类，仅 forward 末尾改为残差相加。
"""
import torch
import torch.nn as nn

from .pillar_attention import PillarAttention


class PillarAttentionRes(PillarAttention):
    """PillarAttention + ResNet 短接（保留 PillarVFE 原始几何特征）。

    新增 cfg:
      RES_SHORTCUT (bool, 默认 True): 是否启用残差短接；False 则退化为 a1 行为。
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, **kwargs)
        self.res_shortcut = self.model_cfg.get('RES_SHORTCUT', True)
        # 短接通道对齐：输入通道 -> attn_channels。
        # 同通道（当前 cfg ATTN_CHANNELS=32=VFE NUM_FILTERS=32）时为 Identity，
        # 不引入额外参数，退化为纯恒等短接。
        self.short_cut = (
            nn.Identity() if input_channels == self.attn_channels
            else nn.Linear(input_channels, self.attn_channels)
        )

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']  # (num_pillars, C_in)
        identity = self.short_cut(pillar_features)       # (num_pillars, C_attn) 残差支路

        # 复用父类 Transformer block：attn + FFN + norm（含其内部两个小残差），
        # 但父类 forward 末尾整体覆盖 pillar_features。这里先备份原始、再调父类、最后残差相加。
        super().forward(batch_dict)                       # 写入 batch_dict['pillar_features'] = block_out
        block_out = batch_dict['pillar_features']         # (num_pillars, C_attn)

        if self.res_shortcut and identity.shape == block_out.shape:
            block_out = identity + block_out              # ResNet 短接：B1 + TransformerBlock(B1)

        batch_dict['pillar_features'] = block_out
        return batch_dict
