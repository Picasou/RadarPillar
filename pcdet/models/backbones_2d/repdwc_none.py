"""RPiN：RepBlock 多尺度 → ConvTranspose deblock → concat。

对齐 BaseBEVBackbone 的多尺度融合拓扑，使 RepDWC 实验组（b5-b9/n4）与
standard 组（b1-b4）仅在 block 类型（RepBlock vs 标准 3×3 Conv）上不同，
构成公平对照。RepDWCBackbone 零改动，本类在其 list 输出之上加 deblock+concat。
"""
import torch
import torch.nn as nn

from .rep_dwc import RepDWCBackbone


class RepDWCNoneBackbone(nn.Module):
    """RepBlock 多尺度 → ConvTranspose deblock → concat（对齐 BaseBEVBackbone）。"""

    def __init__(self, model_cfg, input_channels: int = 32):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = RepDWCBackbone(model_cfg, input_channels)  # 返回 list[Tensor]

        out_channels = list(model_cfg.OUT_CHANNELS)
        upsample_strides = list(model_cfg.UPSAMPLE_STRIDES)
        num_upsample_filters = list(model_cfg.NUM_UPSAMPLE_FILTERS)
        assert len(num_upsample_filters) == self.backbone.num_outputs, \
            'NUM_UPSAMPLE_FILTERS 数量须等于 NUM_OUTPUTS'
        assert len(num_upsample_filters) == len(upsample_strides) == len(out_channels)

        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[i], num_upsample_filters[i],
                                   upsample_strides[i], stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(num_upsample_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
            for i in range(len(num_upsample_filters))
        ])
        self.num_bev_features = int(sum(num_upsample_filters))

    def forward(self, data_dict):
        outs = self.backbone(data_dict)                      # [160×160, 80×80, 40×40]
        ups = [db(o) for db, o in zip(self.deblocks, outs)]  # 各上采样到 160×160
        data_dict['spatial_features_2d'] = torch.cat(ups, dim=1)
        return data_dict
