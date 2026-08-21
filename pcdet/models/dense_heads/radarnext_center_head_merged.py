"""RPiN head ablation ④：RadarNeXtCenterHeadMerged（回归合并）。

hm 分支独立（hidden + final C=num_cls），几何回归四头（reg/height/dim/rot）并成一个
merged 分支（hidden + final 8ch），forward 按列拆回 reg/height/dim/rot 同名键，
父类 loss/get_targets/predict 无需改动。列序与父类 anno_box 拼接序一致
（reg, height, dim, rot；MSR 无 vel）。
"""
import copy

from torch import nn

from .radarnext_center_head import RadarNeXtCenterHead

# merged 分支列序 = 父类 loss_by_feat 的 cat 序（vel 不支持，遇到即报错）
MERGE_ORDER = ('reg', 'height', 'dim', 'rot')


class MergedSepHead(nn.Module):
    """hm 独立分支 + 几何 merged 分支，输出键与 SepHead 兼容。"""

    def __init__(self, in_channels, num_cls, common_heads, num_hm_conv,
                 init_bias, final_kernel, bn=True, merge_order=None):
        super(MergedSepHead, self).__init__()
        self.merge_order = tuple(merge_order) if merge_order else MERGE_ORDER
        unsupported = set(common_heads) - set(self.merge_order)
        assert not unsupported, f'MergedSepHead 不支持的头: {unsupported}'

        # hm 分支：num_hm_conv-1 层 hidden + final(num_cls)，结构对齐 SepHead 的 hm
        hm_fc = nn.Sequential()
        for _ in range(num_hm_conv - 1):
            hm_fc.append(nn.Conv2d(in_channels, in_channels,
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True))
            if bn:
                hm_fc.append(nn.BatchNorm2d(in_channels))
            hm_fc.append(nn.ReLU())
        hm_fc.append(nn.Conv2d(in_channels, num_cls, kernel_size=final_kernel,
                               stride=1, padding=final_kernel // 2, bias=True))
        hm_fc[-1].bias.data.fill_(init_bias)
        self.hm_fc = hm_fc

        # merged 分支：1 层 hidden + final(sum C)，通道按 merge_order 排列
        self.merge_channels = [int(common_heads[k][0]) for k in self.merge_order]
        merged_out = sum(self.merge_channels)
        merged_fc = nn.Sequential()
        merged_fc.append(nn.Conv2d(in_channels, in_channels,
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True))
        if bn:
            merged_fc.append(nn.BatchNorm2d(in_channels))
        merged_fc.append(nn.ReLU())
        merged_fc.append(nn.Conv2d(in_channels, merged_out, kernel_size=final_kernel,
                                   stride=1, padding=final_kernel // 2, bias=True))
        self.merged_fc = merged_fc

    def forward(self, x):
        """
        双分支预测: 输出 dict{hm, reg, ...几何键}，merged 通道按列拆键
        """
        out = {'hm': self.hm_fc(x)}
        merged = self.merged_fc(x)
        i = 0
        for name, c in zip(self.merge_order, self.merge_channels):
            out[name] = merged[:, i:i + c]
            i += c
        return out


class RadarNeXtCenterHeadMerged(RadarNeXtCenterHead):
    """回归合并变体：tasks 内 SepHead 替换为 MergedSepHead。"""

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size,
                 point_cloud_range, predict_boxes_when_training=True):
        super(RadarNeXtCenterHeadMerged, self).__init__(
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
                              init_bias, final_kernel)
            )
        self.init_weights()
