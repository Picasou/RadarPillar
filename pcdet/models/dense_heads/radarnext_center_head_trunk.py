"""RPiN head ablation ⑤：RadarNeXtCenterHeadTrunk（全共享 trunk）。

1 层全头共用 hidden（ConvBlock 64→64 3×3）+ 每头仅 1 个 1×1 final
（hm/reg/height/dim/rot），输出键与 SepHead 兼容，父类
loss/get_targets/predict 无需改动。
"""
import copy

from torch import nn

from .radarnext_center_head import ConvBlock, RadarNeXtCenterHead


class TrunkSepHead(nn.Module):
    """共享 trunk + 各头 1×1 final。"""

    def __init__(self, in_channels, num_cls, common_heads, num_hm_conv_unused,
                 init_bias, **kwargs):
        super(TrunkSepHead, self).__init__()
        # 共用 hidden：ConvBlock(in→in, 3×3)（in=SHARE_CONV_CHANNEL，与头同宽）
        self.trunk = ConvBlock(in_channels, in_channels, kernel_size=3)

        # 各头 final（1×1）：hm + common_heads 全体
        finals = dict(hm=num_cls)
        finals.update({k: int(v[0]) for k, v in common_heads.items()})
        self.final_names = list(finals.keys())
        for name, c in finals.items():
            final = nn.Conv2d(in_channels, c, kernel_size=1, stride=1,
                              padding=0, bias=True)
            if name == 'hm':
                final.bias.data.fill_(init_bias)
            self.__setattr__(name, final)

    def forward(self, x):
        """
        共享 trunk 预测: 输出 dict{hm, reg, height, dim, rot}
        """
        t = self.trunk(x)
        return {name: self.__getattr__(name)(t) for name in self.final_names}


class RadarNeXtCenterHeadTrunk(RadarNeXtCenterHead):
    """全共享 trunk 变体：tasks 内 SepHead 替换为 TrunkSepHead。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHeadTrunk, self).__init__(
            model_cfg, input_channels, num_class, class_names, grid_size,
            point_cloud_range, predict_boxes_when_training)

        cfg = self.model_cfg
        share_conv_channel = int(cfg.get('SHARE_CONV_CHANNEL', 64))
        init_bias = float(cfg.get('INIT_BIAS', -2.19))

        self.tasks = nn.ModuleList()
        for (num_cls, stride) in zip(self.num_classes, self.strides):
            self.tasks.append(
                TrunkSepHead(share_conv_channel, num_cls,
                             copy.deepcopy(self.common_heads), 0, init_bias)
            )
        self.init_weights()
