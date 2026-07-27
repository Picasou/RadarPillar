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
    """A2：纯通道注意力（BEV 域）。scatter→BEV-GAP（逐样本）→SE 门→scatter 回 pillar。

    squeeze 用 `bev.mean(dim=(2,3))`（≡ RadarNeXt SEBlock 的 F.avg_pool2d），是逐样本
    (B,C) 门——与 A3(SEDWConv) 对齐，输出不随 batch 组合变化。A3 = 本模块 + DWConv 分支 + 残差。
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        grid_size = kwargs.get('grid_size')
        assert grid_size is not None, 'SEBlock 需要 grid_size（detector build 会传）'
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])
        attn_channels = int(model_cfg.get('ATTN_CHANNELS', input_channels))
        reduction = int(model_cfg.get('REDUCTION', 4))
        self.num_point_features = attn_channels
        # 输入与注意力通道不同时做线性对齐（与 PillarAttention 一致做法）
        self.pre_mlp = (nn.Linear(input_channels, attn_channels, bias=False)
                        if input_channels != attn_channels else nn.Identity())
        self.bottleneck = build_se_bottleneck(attn_channels, reduction)

    def forward(self, batch_dict):
        pf = batch_dict['pillar_features']           # (M, C_in)
        coords = batch_dict['voxel_coords']           # (M, 4) [b, z, y, x]
        batch_size = int(coords[:, 0].max().int().item()) + 1

        # scatter → BEV（z 维恒 0，(b,y,x) 唯一 — voxelization 保证），与 SEDWConv 同款
        bev = pf.new_zeros((batch_size, self.num_point_features, self.ny, self.nx))
        b = coords[:, 0].long()
        y = coords[:, 2].long().clamp(0, self.ny - 1)
        x = coords[:, 3].long().clamp(0, self.nx - 1)
        bev[b, :, y, x] = self.pre_mlp(pf)            # (M, C_attn)

        w = self.bottleneck(bev.mean(dim=(2, 3)))     # (B, C_attn) 逐样本门 ≡ avg_pool2d
        out = bev * w.unsqueeze(-1).unsqueeze(-1)     # (B, C_attn, H, W) 通道重标定（无残差）
        batch_dict['pillar_features'] = out[b, :, y, x]   # 还原 (M, C_attn)
        return batch_dict
