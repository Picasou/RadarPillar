"""PP*/head_2d 共享的 standard 多尺度 backbone（无 deblock），输出 large→small 列表。
PPFPN/PPMDFEN 共用，RepDWCNone 走 rep_dwc 路径不经过此处。
"""
import torch
import torch.nn as nn


class StandardMultiScale(nn.Module):
    """BaseBEVBackbone 的 blocks 部分去掉 deblock，按 large→small 顺序产出多尺度 list。

    与 BaseBEVBackbone 对齐：ZeroPad2d(1) + Conv2d(k=3, stride=s, padding=0) + BN + ReLU，
    再 k-1 层 k=3 p=1 Conv-BN-ReLU。stride>1 时严格 pad-conv（与 base 一致）。
    """

    def __init__(self, model_cfg, input_channels: int):
        super().__init__()
        self.model_cfg = model_cfg
        layer_nums = list(model_cfg.LAYER_NUMS)
        layer_strides = list(model_cfg.LAYER_STRIDES)
        num_filters = list(model_cfg.NUM_FILTERS)
        assert len(layer_nums) == len(layer_strides) == len(num_filters), \
            f'LAYER_NUMS/STRIDES/NUM_FILTERS 长度不一致: {layer_nums}/{layer_strides}/{num_filters}'

        c_in_list = [input_channels, *num_filters[:-1]]
        blocks = []
        for i in range(len(layer_nums)):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[i], num_filters[i], kernel_size=3,
                          stride=layer_strides[i], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[i]):
                layers.extend([
                    nn.Conv2d(num_filters[i], num_filters[i], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """x: (B, C_in, H, W) → list[Tensor] 大→小顺序，元素个数 == len(NUM_FILTERS)。"""
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs
