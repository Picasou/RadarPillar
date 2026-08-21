"""RPiN head ablation ⑤ 2D 版：RadarNeXtCenterHead2DTrunk（NoZ + 全共享 trunk）。

NoZ 真 2D 基础上，全头（hm/reg/dim/rot）共用 1 层 hidden 64→64 3×3，
每头仅 1×1 final。TrunkSepHead 复用 3D 版（finals 由 common_heads 决定，
NoZ 版即 hm/reg/dim/rot）。loss/get_targets/predict 全复用 NoZ。
"""
import copy

from torch import nn

from .radarnext_center_head_2d_noz import RadarNeXtCenterHead2DNoZ
from .radarnext_center_head_trunk import TrunkSepHead


class RadarNeXtCenterHead2DTrunk(RadarNeXtCenterHead2DNoZ):
    """2D NoZ 全共享 trunk 变体。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHead2DTrunk, self).__init__(
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
