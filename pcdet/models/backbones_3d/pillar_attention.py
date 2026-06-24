import torch
import torch.nn as nn


class PillarAttention(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = self.model_cfg.ATTN_CHANNELS
        # 修改 1：按论文设置统一通道数（Uniform Scaling）
        self.attn_channels = self.model_cfg.get('ATTN_CHANNELS', input_channels)
        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.get('DROPOUT', 0.0)
        # FFN hidden 维度：由配置 FFN_CHANNELS 驱动；缺省值为 attn_channels * 2（向后兼容旧配置）。
        # 论文图 3 中 FFN hidden 标注为 E（主配置 E=32）。
        self.ffn_hidden = self.model_cfg.get('FFN_CHANNELS', self.attn_channels * 2)

        # 若输入通道数与 attention 通道数不同，则用单层线性映射对齐
        self.pre_mlp = nn.Linear(input_channels, self.attn_channels) if input_channels != self.attn_channels else nn.Identity()

        # 修改 2：MultiheadAttention —— batch_first=True 可显著加速
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(self.attn_channels)

        # FFN hidden 维度由 FFN_CHANNELS 配置控制；论文主配置使用 E=hidden=32。
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_channels, self.ffn_hidden),
            nn.GELU(),
            nn.Linear(self.ffn_hidden, self.attn_channels),
        )
        self.norm2 = nn.LayerNorm(self.attn_channels)


    def forward(self, batch_dict):
            pillar_features = batch_dict['pillar_features'] # (num_pillars, C)
            coords = batch_dict['voxel_coords']             # (num_pillars, 4) [batch_idx, z, y, x]

            batch_size = coords[:, 0].max().int().item() + 1

            # 修改 4：摆脱 for 循环，改用 masked padding 方案
            # 统计每个 batch 内的 pillar 数量
            pillar_counts = []
            for b in range(batch_size):
                pillar_counts.append((coords[:, 0] == b).sum().item())

            max_pillars = max(pillar_counts)

            # 创建空模板 (Batch, Max_Pillar, Channels)
            padded_features = torch.zeros((batch_size, max_pillars, pillar_features.shape[-1]),
                                        device=pillar_features.device)

            # 稀疏性 mask：值为 True 的位置视为“空”，attention 将其忽略
            key_padding_mask = torch.ones((batch_size, max_pillars), dtype=torch.bool,
                                        device=pillar_features.device)

            # 将平铺列表形式的数据转为 batch 格式（该操作在 GPU 上很快）
            for b in range(batch_size):
                mask = coords[:, 0] == b
                num_p = pillar_counts[b]
                padded_features[b, :num_p] = pillar_features[mask]
                key_padding_mask[b, :num_p] = False # 将已被填充的位置从 mask 中排除

            # 修改 5：单次完成 ATTENTION 计算
            x = self.pre_mlp(padded_features)

            # key_padding_mask：确保模型只关注真实存在的 pillar
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm1(x + attn_out)

            # Feed Forward Network
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            # 修改 6：将结果还原为 OpenPCDet 期望的 (num_pillars, C) 形式
            # 去掉 padding，恢复为 OpenPCDet 所期望的 (num_pillars, C) 形状
            updated_features = []
            for b in range(batch_size):
                updated_features.append(x[b, :pillar_counts[b]])

            batch_dict['pillar_features'] = torch.cat(updated_features, dim=0)
            return batch_dict
