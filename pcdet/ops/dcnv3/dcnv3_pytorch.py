"""Pure-pytorch DCNv3 (Deformable Convolution v3) implementation.

Ported verbatim (numerics-identical) from
``projects/RadarNeXt/radarnext/DeformFFN.py``'s ``DCNv3_pytorch`` class and
its helpers (``dcnv3_core_pytorch``, ``_get_reference_points``,
``_generate_dilation_grids``, ``CenterFeatureScaleModule``,
``build_norm_layer``, ``build_act_layer``).

Only the mmdet3d registry / ``import DCNv3`` machinery is dropped; the
forward numerics (grid_sample-based deformable sampling) are preserved
bit-for-bit so that ``tests/parity/test_parity_mdfen.py`` can reproduce the
original's output exactly when the original is given the SAME pure-pytorch
DCNv3 (via the parity stub in ``_canary.py``).

Why this file exists:
    The CUDA ``DCNv3`` extension (InternImage ``ops_dcnv3``) requires an
    out-of-tree C++/CUDA build that fails under base py3.12/cu124. Per the
    Task 7 brief's §6 exhaustion chain, this pure-pytorch implementation is
    the never-fail fallback — it runs in any torch environment and is the
    primary DCNv3 used by the MDFEN port. The CUDA op is an OPTIONAL
    accelerator that is NOT required for correctness or parity.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    """Build a norm layer matching the InternImage/RadarNeXt contract.

    ``'BN'`` wraps a channels-last -> channels-first permute, BatchNorm2d,
    channels-first -> channels-last permute. ``'LN'`` uses LayerNorm directly
    on the channel dim.
    """
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w,
                          dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device),
        indexing='ij')
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w,
                             dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device),
        indexing='xy')
    # The original RadarNeXt code uses ``points_list.extend([x / W_, y / H_])``
    # where ``x`` comes from the first linspace arg above (the "w" axis) and
    # ``y`` from the second (the "h" axis). ``torch.meshgrid(indexing='xy')``
    # reproduces the legacy default (no indexing=) shape used by the original.
    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    """Pure-pytorch core DCNv3 sampling (grid_sample-based).

    Faithful port of ``DeformFFN.dcnv3_core_pytorch``. Used by
    ``DCNv3_pytorch.forward``. Inputs/outputs are channels-last:
        input:  (N, H_in+2*pad, W_in+2*pad, C)
        offset: (N, H_out, W_out, group * P * 2)
        mask:   (N, H_out, W_out, group * P)
        output: (N, H_out, W_out, C)
    """
    # NOTE: the original pads as [0, 0, pad_h, pad_h, pad_w, pad_w] which
    # corresponds to (W: none, H: both, W: both) under F.pad's
    # "last-dim-first" convention applied to a (N,H,W,C) tensor — i.e. the
    # spatial H and W dims get padded symmetrically. Preserve verbatim.
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w,
        pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group,
        input.device)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group * kernel_h * kernel_w).to(input.device)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels
    # -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).\
        reshape(N_ * group, group_channels, H_in, W_in)
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2
    # -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros',
        align_corners=False)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_
    # -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).\
        reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(
        N_, group * group_channels, H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


class DCNv3_pytorch(nn.Module):
    """Pure-pytorch DCNv3 module (no CUDA extension).

    Verbatim port of ``DeformFFN.DCNv3_pytorch``. The forward contract is
    channels-last: input ``(N, H, W, C)`` -> output ``(N, H, W, C)``.

    Defaults match the original (GELU + LN norm, center_feature_scale=False).
    """

    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            # 严格对齐论文原工程(DeformLayer 显式 offset_scale=2.0, DCNv3.py:94,168)
            offset_scale=2.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1
            # -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x
