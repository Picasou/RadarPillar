"""RPiN 阶段1 A3 候选：SE + DWConv（深度可分离卷积）。
复用 SEBlock.build_se_bottleneck；DW_KERNEL=3 默认。
契约（BACKBONE_3D 槽）：`__init__(model_cfg, input_channels, **kwargs)`，set `num_point_features`。
"""
import torch
import torch.nn as nn

from .se_block import build_se_bottleneck


class SEDWConv(nn.Module):
    """A3：DWConv(DW_KERNEL) + BN + ReLU + SE bottleneck 门控 + 残差，BEV 域运算。"""

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        grid_size = kwargs.get('grid_size')
        assert grid_size is not None, 'SEDWConv 需要 grid_size（detector build 会传）'
        self.nx, self.ny = int(grid_size[0]), int(grid_size[1])

        attn_channels = int(model_cfg.get('ATTN_CHANNELS', input_channels))
        reduction = int(model_cfg.get('REDUCTION', 4))
        dw_kernel = int(model_cfg.get('DW_KERNEL', 3))
        padding = dw_kernel // 2

        self.num_point_features = attn_channels
        self.pre_mlp = (nn.Linear(input_channels, attn_channels, bias=False)
                        if input_channels != attn_channels else nn.Identity())

        self.dwconv = nn.Conv2d(attn_channels, attn_channels, kernel_size=dw_kernel,
                                padding=padding, groups=attn_channels, bias=False)
        self.pwconv = nn.Conv2d(attn_channels, attn_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(attn_channels)
        self.act = nn.ReLU(inplace=True)
        self.bottleneck = build_se_bottleneck(attn_channels, reduction)

    def forward(self, batch_dict):
        pf = batch_dict['pillar_features']           # (M, C_in)
        coords = batch_dict['voxel_coords']           # (M, 4) [b, z, y, x]
        batch_size = int(coords[:, 0].max().int().item()) + 1

        # scatter → BEV（z 维恒 0，(b,y,x) 唯一 — voxelization 保证）
        bev = pf.new_zeros((batch_size, self.num_point_features, self.ny, self.nx))
        b = coords[:, 0].long()
        y = coords[:, 2].long().clamp(0, self.ny - 1)
        x = coords[:, 3].long().clamp(0, self.nx - 1)
        bev[b, :, y, x] = self.pre_mlp(pf)            # 高级索引结果 (M, C_attn)

        feat = self.act(self.bn(self.pwconv(self.dwconv(bev))))
        w = self.bottleneck(feat.mean(dim=(2, 3)))    # (B, C_attn)
        out = bev + feat * w.unsqueeze(-1).unsqueeze(-1)
        batch_dict['pillar_features'] = out[b, :, y, x]
        return batch_dict
