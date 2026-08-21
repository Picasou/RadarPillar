"""RPiN head ablation ④ 2D 版：RadarNeXtCenterHead2DMerged（NoZ + 回归合并）。

NoZ 真 2D 基础上，hm 分支独立，几何回归三头（reg/dim/rot）并成一个 merged
分支（6ch = dx,dy + log l,w + sin,cos），forward 按列拆回同名键。
列序与 NoZ 的 anno_box 拼接序一致。
"""
import copy

from torch import nn

from .radarnext_center_head_2d_noz import RadarNeXtCenterHead2DNoZ
from .radarnext_center_head_merged import MergedSepHead

MERGE_ORDER_2D = ('reg', 'dim', 'rot')


class RadarNeXtCenterHead2DMerged(RadarNeXtCenterHead2DNoZ):
    """2D NoZ 回归合并变体：tasks 内 SepHead 替换为 2D 版 MergedSepHead。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHead2DMerged, self).__init__(
            model_cfg, input_channels, num_class, class_names, grid_size,
            point_cloud_range, predict_boxes_when_training)

        cfg = self.model_cfg
        share_conv_channel = int(cfg.get('SHARE_CONV_CHANNEL', 64))
        num_hm_conv = int(cfg.get('NUM_HM_CONV', 2))
        init_bias = float(cfg.get('INIT_BIAS', -2.19))
        final_kernel = int(cfg.get('FINAL_KERNEL', 3))

        self.tasks = nn.ModuleList()
        for (num_cls, stride) in zip(self.num_classes, self.strides):
            self.tasks.append(
                MergedSepHead(share_conv_channel, num_cls,
                              copy.deepcopy(self.common_heads), num_hm_conv,
                              init_bias, final_kernel, merge_order=MERGE_ORDER_2D)
            )
        self.init_weights()
