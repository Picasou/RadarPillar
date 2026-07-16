"""Re-parameterizable common blocks (pure PyTorch port from RadarNeXt).

Original source: projects/RadarNeXt/radarnext/common.py
Only the RepDWC-backbone-relevant pieces are ported here: ``ConvBNReLU``,
``Transpose`` and ``RepBlock``. The DCN / DeformFFN / BiFusion parts of the
original ``common.py`` belong to the neck and are out of scope for Task 2.

Logic is preserved verbatim from the original — ``RepBlock`` builds
depthwise-then-pointwise ``MobileOneBlock`` stages and the ``n=1`` short-circuit
(``self.block is None``) must be kept for parameter-count parity.
"""

import torch
from torch import nn

from .mobileone_blocks import MobileOneBlock


class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class RepBlock(nn.Module):
    '''
        RepBlock fuses the concatenated feature maps with the MobileOne Block (dwconv + overparameterized training)
        Fusing the multi-scale feature maps along with the channel dimension
        args:
            in_channels (int): the channels of the concatenated feature map
            out_channels (int): the channels of the outputs
            kernel_size (int): the kernel size of convolution layers in each RepBlock
            stride (int): the stride of fusion layer in this RepBlock
            n (int): the total number of stacked MobileOne Blocks (including the fusion block)
            inference_mode (bool): Whether to define a single-path model for inference
            use_se (bool): Whether to use SE-ReLU as the activation function
            num_conv_branches (int): the number of convolutional layers stacked on the rbr_conv branch
            use_dwconv (bool): Whether to use Depthwise Separate Convolution
            use_normconv (bool): Whether to use normal Convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 n=1,           # the depth of RepBlock
                 inference_mode = False,
                 use_se = False,
                 num_conv_branches = 1,
                 use_dwconv = True,
                 use_normconv = False):
        super().__init__()

        self.inference_mode = inference_mode
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        assert not use_normconv and use_dwconv, 'only one type of convolutional layers can be built'
        assert use_normconv or use_dwconv, 'must choose one type for convolutional layers'
        self.use_dwconv = use_dwconv
        self.use_normconv = use_normconv

        self.fuse = self._make_stage(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.block = nn.Sequential(*(self._make_stage(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1) for _ in range(n - 1))) if n > 1 else None

    def _make_stage(self,
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=1,
                    padding=None):
        if padding is None:
            padding = kernel_size // 2

        if self.use_dwconv:
            # Depthwise conv
            block = [MobileOneBlock(in_channels=in_planes,
                                    out_channels=in_planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_planes,
                                    inference_mode=self.inference_mode,
                                    use_se=self.use_se,
                                    num_conv_branches=self.num_conv_branches)]

            # Pointwise conv
            block.append(MobileOneBlock(in_channels=in_planes,
                                        out_channels=planes,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=1,
                                        inference_mode=self.inference_mode,
                                        use_se=self.use_se,
                                        num_conv_branches=self.num_conv_branches))
            return nn.Sequential(*block)
        elif self.use_normconv:
            block = MobileOneBlock(in_channels=in_planes,
                                   out_channels=planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=1,
                                   inference_mode=self.inference_mode,
                                   use_se=self.use_se,
                                   num_conv_branches=self.num_conv_branches)
            return block
        else:
            raise ValueError(f'one of convolution types should be chosen, but use_normconv: {self.use_normconv}, and use_dwconv: {self.use_dwconv}.')

    def forward(self, x):
        x = self.fuse(x)
        if self.block is not None:
            x = self.block(x)
        return x
