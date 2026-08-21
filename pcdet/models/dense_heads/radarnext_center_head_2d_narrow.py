"""RPiN head ablation ③ 2D 版：RadarNeXtCenterHead2DNarrow（NoZ + 降宽）。

NoZ 真 2D 基础（无 height 头、dim 2ch、6-cat 监督）上收窄通道：
SHARE_CONV_CHANNEL / HEAD_CONV（如 64→32），头数结构不变。
继承 NoZ 的 target/loss/predict 全链路，仅按 HEAD_CONV 重建 tasks。
"""
import copy

from torch import nn

from .radarnext_center_head import SepHead
from .radarnext_center_head_2d_noz import RadarNeXtCenterHead2DNoZ


class RadarNeXtCenterHead2DNarrow(RadarNeXtCenterHead2DNoZ):
    """2D NoZ 降宽变体：head_conv 接 cfg（HEAD_CONV）。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHead2DNarrow, self).__init__(
            model_cfg, input_channels, num_class, class_names, grid_size,
            point_cloud_range, predict_boxes_when_training)

        cfg = self.model_cfg
        share_conv_channel = int(cfg.get('SHARE_CONV_CHANNEL', 64))
        head_conv = int(cfg.get('HEAD_CONV', 64))
        num_hm_conv = int(cfg.get('NUM_HM_CONV', 2))
        init_bias = float(cfg.get('INIT_BIAS', -2.19))
        final_kernel = int(cfg.get('FINAL_KERNEL', 3))

        # 按 HEAD_CONV 重建 tasks（父类版本 head_conv 固定 64）；common_heads 已是 NoZ 版（无 height, dim 2ch）
        self.tasks = nn.ModuleList()
        for (num_cls, stride) in zip(self.num_classes, self.strides):
            heads = copy.deepcopy(self.common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=final_kernel,
                        head_conv=head_conv)
            )
        self.init_weights()
