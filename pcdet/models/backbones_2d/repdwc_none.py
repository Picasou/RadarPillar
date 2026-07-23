"""RPiN 阶段3 N-4：RepDWC 末层直出（无 neck）。复用 RepDWCBackbone 取 outs[-1]。"""
import torch
import torch.nn as nn

from .rep_dwc import RepDWCBackbone


class RepDWCNoneBackbone(nn.Module):
    """Stage 3 N-4：RepDWC 取末层直出。num_bev_features=OUT_CHANNELS[-1]。"""

    def __init__(self, model_cfg, input_channels: int = 32):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = RepDWCBackbone(model_cfg, input_channels)
        self.num_bev_features = int(list(model_cfg.OUT_CHANNELS)[-1])

    def forward(self, data_dict):
        outs = self.backbone(data_dict)   # list[Tensor] large→small
        data_dict['spatial_features_2d'] = outs[-1]
        return data_dict
