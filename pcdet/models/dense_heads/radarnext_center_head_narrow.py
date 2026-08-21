"""RPiN head ablation ③：RadarNeXtCenterHeadNarrow（降宽）。

SHARE_CONV_CHANNEL / HEAD_CONV 同步收窄（如 64→32），SepHead 结构与头数不变。
父类构建 tasks 时未把 head_conv 接到 cfg（写死默认 64），本子类按 HEAD_CONV 重建
tasks；shared_conv 宽度由父类按 SHARE_CONV_CHANNEL 构建，无需重建。
loss/get_targets/predict 全复用父类。
"""
import copy

from torch import nn

from .radarnext_center_head import RadarNeXtCenterHead, SepHead


class RadarNeXtCenterHeadNarrow(RadarNeXtCenterHead):
    """降宽变体：head_conv 接 cfg（HEAD_CONV），其余结构同父类。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHeadNarrow, self).__init__(
            model_cfg, input_channels, num_class, class_names, grid_size,
            point_cloud_range, predict_boxes_when_training)

        cfg = self.model_cfg
        share_conv_channel = int(cfg.get('SHARE_CONV_CHANNEL', 64))
        head_conv = int(cfg.get('HEAD_CONV', 64))
        num_hm_conv = int(cfg.get('NUM_HM_CONV', 2))
        num_corner_hm_conv = int(cfg.get('NUM_CORNER_HM_CONV', 2))
        init_bias = float(cfg.get('INIT_BIAS', -2.19))
        final_kernel = int(cfg.get('FINAL_KERNEL', 3))

        # 按 HEAD_CONV 重建 tasks（父类版本 head_conv 固定 64）
        self.tasks = nn.ModuleList()
        for (num_cls, stride) in zip(self.num_classes, self.strides):
            heads = copy.deepcopy(self.common_heads)
            if self.with_corner:
                heads.update(dict(hm=(num_cls, num_hm_conv),
                                  corner_hm=(1, num_corner_hm_conv)))
            else:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, stride=stride,
                        bn=True, init_bias=init_bias, final_kernel=final_kernel,
                        head_conv=head_conv)
            )
        self.init_weights()
