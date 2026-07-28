"""PillarAttentionRes-B：方案 B（ReZero 零初始化 ffn，RPiN 阶段1）。

动机（来自 experiments/stage1/方案档位.md §4）：
父类 PillarAttentionRes 的 ffn = Sequential(Linear, GELU, Linear)。
epoch 0 时 ffn[2] (第二个 Linear) 的 weight/bias 若服从随机初始化，
会输出非零 Wx+b，可能与 attn 输出叠加后污染 PillarVFE 原始特征。

方案 B 修法：将 ffn[2] 的 weight 与 bias 初始化为 0（前向公式 y = Wx+b → 0），
使 epoch 0 时 ffn 输出严格为 0，
    block_out = norm1(x + attn_out), x 为父类 forward 的最终输出
attn 路径在 epoch 0 不污染 PillarVFE 输出。
后续 epoch 一旦 ffn 学到非零权重 → 机制失效，仅 epoch 0 干净。

注意 ReZero 初始化只改 data（原地 zeros_），不改变 state_dict 结构与层形状。
"""
import torch.nn as nn

from .pillar_attention_res import PillarAttentionRes


class PillarAttentionResB(PillarAttentionRes):
    """方案 B：ReZero 零初始化 ffn[2]（RPiN 阶段1 a4 失败对症方案）。

    将父类 fnn[2]（第二个 Linear 层）的 weight 与 bias 初始化为 0：
      - epoch 0 时 ffn 输出严格为 0（y = 0·x + 0 = 0）
      - block_out = norm1(x + attn_out), x 为父类 forward 内部剩余结构
      - 给 PillarVFE 一个仅 epoch 0 不被 attention 路径污染的起点
      - 后续 epoch ffn 学到非零 W → 机制失效

    严格保证：
      - 0 新增 params（原地修改 ffn[2] 的 weight/bias.data）
      - 0 FLOPs 增量（仅初始化阶段修改 data）

    cfg 不变（与 a4.yaml 兼容），通过 BACKBONE_3D.NAME=PillarAttentionResB 启用。
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels, **kwargs)
        # 方案 B：ReZero 初始化 ffn[2]，epoch 0 时 ffn 输出严格为 0
        # self.ffn = Sequential(Linear, GELU, Linear) -> ffn[2] = 第二个 Linear
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, batch_dict):
        # forward 逻辑完全继承父类（PillarAttentionRes = a4 原状，无 LN-Post 改动）
        return super().forward(batch_dict)
