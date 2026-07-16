"""RepDWC backbone — OpenPCDet port of RadarNeXt's RepDWC.

Original source: projects/RadarNeXt/radarnext/rep_dwc.py (MMDetection3D).
Re-implemented as a pure-torch ``nn.Module`` following OpenPCDet's
``BaseBEVBackbone`` conventions:

* constructor ``__init__(self, model_cfg, input_channels=32)``
  (``model_cfg`` is an EasyDict coming from the YAML ``BACKBONE_2D`` block);
* ``forward`` accepts either a ``data_dict`` (with ``spatial_features``) or a
  raw tensor, matching OpenPCDet's call style, and returns the multi-scale
  outputs as a ``list[Tensor]`` (OpenPCDet multi-scale convention) instead of a
  tuple.

The per-stage block assembly (``RepBlock`` stack, ``num_outputs`` last stages
returned) is preserved verbatim from the original — only the I/O shape and the
registry/init wiring changed.
"""

import torch
from torch import Tensor
from torch import nn as nn

from .rep_common import RepBlock


class RepDWCBackbone(nn.Module):
    """Re-parameterizable backbone with MobileOne's Architecture (OpenPCDet port).

    Multi-scale BEV backbone. Training-time it is multi-branched (depthwise
    MobileOneBlock + pointwise MobileOneBlock per stage); at inference the
    branches can be fused via :func:`reparameterize_model`.

    Args:
        model_cfg (EasyDict): YAML ``BACKBONE_2D`` block. Expected keys:
            ``OUT_CHANNELS`` (list[int]), ``LAYER_NUMS`` (list[int]),
            ``LAYER_STRIDES`` (list[int]), ``NUM_OUTPUTS`` (int, default 3),
            ``INFERENCE_MODE`` (bool, default False), ``USE_SE`` (bool, False),
            ``NUM_CONV_BRANCHES`` (int, 1), ``USE_DWCONV`` (bool, True),
            ``USE_NORMCONV`` (bool, False).
        input_channels (int): Number of input BEV feature channels. Per the
            project audit (M4) this is ``NUM_BEV_FEATURES=32``, NOT the 64 found
            in the original RadarNeXt MMDet config.

    Forward returns a ``list[Tensor]`` of the last ``num_outputs`` stages, e.g.
    ``[(B,64,160,160),(B,128,80,80),(B,256,40,40)]`` for the FPN-variant config
    fed a ``(B,32,320,320)`` input.
    """

    def __init__(self, model_cfg, input_channels: int = 32):
        super(RepDWCBackbone, self).__init__()
        self.model_cfg = model_cfg

        # --- read hyperparameters from the EasyDict (YAML BACKBONE_2D block) ---
        # NOTE: input channels come from the constructor (NUM_BEV_FEATURES=32, audit M4),
        # NOT from model_cfg — the original MMDet config's in_channels=64 does not apply here.
        in_channels = input_channels
        out_channels = list(model_cfg.OUT_CHANNELS)
        layer_nums = list(model_cfg.LAYER_NUMS)
        layer_strides = list(model_cfg.LAYER_STRIDES)
        num_outputs = model_cfg.get('NUM_OUTPUTS', 3)
        inference_mode = model_cfg.get('INFERENCE_MODE', False)
        use_se = model_cfg.get('USE_SE', False)
        num_conv_branches = model_cfg.get('NUM_CONV_BRANCHES', 1)
        use_normconv = model_cfg.get('USE_NORMCONV', False)
        use_dwconv = model_cfg.get('USE_DWCONV', True)

        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)
        assert not use_normconv and use_dwconv, 'only one type of convolutional layers can be built'
        assert use_normconv or use_dwconv, 'must choose one type for convolutional layers'
        self.num_outputs = num_outputs
        # Exposed for detector assembly (mirrors BaseBEVBackbone.num_bev_features semantics,
        # though this backbone emits multi-scale features rather than a single fused map).
        self.num_bev_features = in_channels

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            blocks.append(
                RepBlock(
                    in_channels=in_filters[i],
                    out_channels=out_channels[i],
                    kernel_size=3,
                    stride=layer_strides[i],
                    n=layer_num,
                    inference_mode=inference_mode,
                    use_se=use_se,
                    num_conv_branches=num_conv_branches,
                    use_dwconv=use_dwconv,
                    use_normconv=use_normconv
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, data_dict_or_tensor):
        """Forward function.

        Args:
            data_dict_or_tensor: either a ``dict``/EasyDict carrying
                ``spatial_features`` (OpenPCDet detector style) or a raw
                ``(N, C, H, W)`` tensor.

        Returns:
            list[torch.Tensor]: the last ``num_outputs`` multi-scale features.
        """
        if isinstance(data_dict_or_tensor, dict):
            x = data_dict_or_tensor['spatial_features']
        else:
            x = data_dict_or_tensor

        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i >= (len(self.blocks) - self.num_outputs):  # only return the outputs of the last num_outputs blocks
                outs.append(x)
        return outs
