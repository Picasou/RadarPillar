"""SECONDFPN neck — pure-torch port of mmdet3d's ``second_fpn.py``.

Original source: mmdet3d/models/necks/second_fpn.py (from the RadarNeXt
mmdet3d vendored tree). Re-implemented without mmcv/mmdet3d dependencies:

* No ``@MODELS.register_module()`` decorator and no ``BaseModule``.
* ``build_conv_layer``/``build_upsample_layer``/``build_norm_layer`` are
  replaced by direct ``nn.Conv2d``/``nn.ConvTranspose2d``/``nn.BatchNorm2d``
  with the same default ``bias`` / BN eps / momentum values the mmdet3d
  builders would produce for ``Conv2d``/``deconv``/``BN`` configs.
* Forward returns a ``list[Tensor]`` of length 1 (the mmdet3d original returns
  ``[out]``); callers that want a single tensor take ``outs[0]`` or rely on
  ``RadarNeXtFPNBackbone`` to do so.

The deblock construction (downsample-vs-upsample branch logic, kernel/stride
selection, conv-vs-deconv choice, BN-eps, inplace ReLU) is preserved verbatim
from the original so that the numerical parity tests (Task 4.5) can reproduce
mmdet3d's output bit-for-bit given the same weights.
"""

import numpy as np
import torch
from torch import nn as nn


class SecondFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet (pure-torch port).

    Mirrors mmdet3d's ``SECONDFPN`` deblock-for-deblock. For each input level
    a *deblock* = ``Sequential(upsample_layer, norm, ReLU(inplace=True))`` is
    built; all deblocks are applied elementwise and their outputs concatenated
    along the channel dim.

    Args:
        in_channels (list[int]): Input channels of each multi-scale feature map.
        out_channels (list[int]): Output channels of each deblock. The fused
            output has ``sum(out_channels)`` channels.
        upsample_strides (list[float]): Per-level stride. Values ``> 1`` use a
            transposed conv (upsample); values ``< 1`` use a stride-``round(1/s)``
            conv (downsample); value ``== 1`` uses a conv iff
            ``use_conv_for_no_stride`` else a kernel-1 transposed conv.
        norm_cfg (dict): ``BN`` config — only ``eps``/``momentum`` honored.
        upsample_cfg (dict): ``deconv`` config — only ``bias`` honored.
        conv_cfg (dict): ``Conv2d`` config — only ``bias`` honored.
        use_conv_for_no_stride (bool): See ``upsample_strides`` above.

    Forward returns ``[out]`` where ``out`` has shape
    ``(N, sum(out_channels), H_target, W_target)`` — all deblock outputs must
    share the same spatial size for the ``cat`` to be valid.
    """

    def __init__(self,
                 in_channels=(128, 128, 256),
                 out_channels=(256, 256, 256),
                 upsample_strides=(1, 2, 4),
                 norm_cfg=None,
                 upsample_cfg=None,
                 conv_cfg=None,
                 use_conv_for_no_stride=False):
        super(SecondFPN, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', eps=1e-3, momentum=0.01)
        if upsample_cfg is None:
            upsample_cfg = dict(type='deconv', bias=False)
        if conv_cfg is None:
            conv_cfg = dict(type='Conv2d', bias=False)

        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = list(in_channels)
        self.out_channels = list(out_channels)
        self.upsample_strides = list(upsample_strides)
        self.use_conv_for_no_stride = use_conv_for_no_stride

        deblocks = []
        for i, out_channel in enumerate(self.out_channels):
            stride = self.upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                # Upsample path: transposed (a.k.a. fractionally-strided) conv.
                # Matches mmdet3d build_upsample_layer(type='deconv', bias=False).
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=self.in_channels[i],
                    out_channels=out_channel,
                    kernel_size=self.upsample_strides[i],
                    stride=self.upsample_strides[i],
                    bias=upsample_cfg.get('bias', False),
                )
            else:
                # Downsample / no-resample path: regular Conv2d with
                # kernel = stride = round(1 / s). For s < 1 this downsamples
                # (e.g. s=0.5 -> k=2,s=2); for s == 1 (with use_conv_for_no_stride)
                # this becomes a 1x1 conv (no resampling).
                stride_int = int(np.round(1 / stride).astype(np.int64))
                upsample_layer = nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride_int,
                    stride=stride_int,
                    bias=conv_cfg.get('bias', False),
                )

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=norm_cfg.get('eps', 1e-3),
                               momentum=norm_cfg.get('momentum', 0.01)),
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (list[torch.Tensor]): Multi-level features, each 4D ``(N,C,H,W)``.

        Returns:
            list[torch.Tensor]: One-element list whose only entry is the
            channel-concatenated fused feature map.
        """
        assert len(x) == len(self.in_channels), \
            f'got {len(x)} inputs for {len(self.in_channels)} deblocks'
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
