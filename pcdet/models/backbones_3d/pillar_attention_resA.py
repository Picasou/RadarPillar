"""PillarAttentionRes-A：方案 A（LN-Post 残差归一化，RPiN 阶段1）。

动机（来自 experiments/stage1/方案档位.md §3）：
父类 PillarAttentionRes.forward 末尾的残差相加：
    block_out = identity + block_out
其中 identity = raw PillarVFE 输出（量级不可控），block_out = TransformerBlock 输出
（已被 norm1/norm2 压回均 0 方 1）。+ 操作在量级错配的两端做加和 →
PillarVFE 几何特征被 TransformerBlock 输出压制。

方案 A 修法：在残差相加前，用 self.norm1 / self.norm2 对 identity（skip 路径）
做归一化，使两端量级一致。注意 norm1/norm2 是 PillarAttention 父类已经存在
的 nn.LayerNorm（已注册到 state_dict），不引入新参数。
"""
import torch.nn as nn

from .pillar_attention_res import PillarAttentionRes


class PillarAttentionResA(PillarAttentionRes):
    """方案 A：LN-Post 残差归一化（RPiN 阶段1 a4 失败对症方案）。

    在父类 PillarAttentionRes.forward 末尾的残差相加前，
    先用 self.norm1 / self.norm2 对 identity（skip 路径）做归一化，
    使其量级与 block_out（attn+ffn 路径）一致，
    避免 epoch 0 时 raw PillarVFE 输出被 TransformerBlock 输出压制。

    严格保证：
      - 0 新增 params（复用 self.norm1, self.norm2）
      - 0 FLOPs 显著增量（LayerNorm = O(N·C) 轻量归一化）
      - 残差加和前，skip 与 attention 输出都被压回均 0 方 1 的同一分布

    cfg 不变（与 a4.yaml 兼容），通过 BACKBONE_3D.NAME=PillarAttentionResA 启用。
    """

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']  # (num_pillars, C_in)
        identity = self.short_cut(pillar_features)       # (num_pillars, C_attn) 残差支路

        # 复用父类 Transformer block：attn + FFN + norm（含其内部两个小残差）
        super().forward(batch_dict)
        block_out = batch_dict['pillar_features']         # (num_pillars, C_attn)

        if self.res_shortcut and identity.shape == block_out.shape:
            # 方案 A：复用父类 norm1/norm2 归一化 identity，使 skip 与 block_out 量级对齐
            # norm1 与 norm2 均来自 PillarAttention 父类，已注册到 state_dict，
            # 无新增参数；归一化使 skip 输出也回到均 0 方 1 分布，避免被 attn 输出压制。
            identity_normed = self.norm2(self.norm1(identity))
            block_out = identity_normed + block_out

        batch_dict['pillar_features'] = block_out
        return batch_dict
