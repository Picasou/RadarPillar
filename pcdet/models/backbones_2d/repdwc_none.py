"""RPiN 阶段3 N-4：RepDWC 首层直出（无 neck）。复用 RepDWCBackbone 取 outs[0]。

设计裁决（RPiN 对抗审查，spatial-fix-audit 维度）：RepDWCBackbone 产出多尺度
[(160,160,64),(80,80,128),(40,40,256)]。「无 neck」= 不做 FPN/MDFEN 融合，须挑 1 层
直出。取 **outs[0]=160×160**（而非计划旧文 outs[-1]=40×40），使 n4 与 n1 基准同处
160×160 分辨率 —— 阶段3「块类型(standard vs repdwc) × neck」消融仅隔离块类型变量，
不混入分辨率 confound；同时避免 40×40 (1.28m/cell) 对 0.8m 行人 anchor 的 sub-cell 退化。
"""
import torch
import torch.nn as nn

from .rep_dwc import RepDWCBackbone


class RepDWCNoneBackbone(nn.Module):
    """Stage 3 N-4：RepDWC 取首层（160×160）直出。num_bev_features=OUT_CHANNELS[0]。"""

    def __init__(self, model_cfg, input_channels: int = 32):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = RepDWCBackbone(model_cfg, input_channels)
        self.num_bev_features = int(list(model_cfg.OUT_CHANNELS)[0])

    def forward(self, data_dict):
        outs = self.backbone(data_dict)   # list[Tensor] 大→小 [160,80,40]
        data_dict['spatial_features_2d'] = outs[0]
        return data_dict
