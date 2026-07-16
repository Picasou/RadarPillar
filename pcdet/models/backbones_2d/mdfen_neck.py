"""MDFEN neck — OpenPCDet port of RadarNeXt's MDFENNeck + DeformLayer.

Original source:
    ``projects/RadarNeXt/radarnext/MDFENNeck.py`` (``MultiMAPFusion``,
    ``MDFENNeck``) and ``projects/RadarNeXt/radarnext/common.py``
    (``DeformLayer``).

Translation points honored (per Task 7 brief, config from
``projects/RadarNeXt/configs/radarnext.py``):
    * ``dcn_layer=False`` -> DCN goes through ``former_deform2`` (PAN top-down
      3rd branch, after concat, before RepBlock); input channel =
      ``channels_list[0] + channels_list[4] = 128``.
    * ``use_ffn=False`` -> ``DeformLayer`` holds a BARE ``DCNv3_pytorch`` (not
      DeformFFN). DCNv3 used is the pure-pytorch variant
      (``pcdet.ops.dcnv3.DCNv3_pytorch``) — the §6 never-fail floor.
    * ``num_repeats=[1,1,1,1]`` -> n=1 makes ``RepBlock.block`` None.
    * ``multi_fusion=True`` + ``fusion_strides=[1,2]`` -> the 3 PAN outputs
      are fused into a single ``(B, 384, H_largest, W_largest)`` tensor.
    * mmdet3d ``@MODELS.register_module`` stripped.

DCNv3 input is channels-last; the module does the permute internally
(matching the original ``DeformLayer.forward``).
"""

import torch
from torch import nn

from pcdet.ops.dcnv3 import DCNv3_pytorch
from .rep_common import ConvBNReLU, Transpose, RepBlock


# --------------------------------------------------------------------------- #
# MultiMAPFusion                                                               #
# --------------------------------------------------------------------------- #
class MultiMAPFusion(nn.Module):
    """Fuse the multi-scale outputs of PANNeck into one map.

    Faithful port of ``MDFENNeck.MultiMAPFusion``. When
    ``len(strides) == len(in_channels)`` each input gets its own
    ``Transpose`` (ConvTranspose2d) deblock; otherwise the first input is
    downsampled with a stride-2 ConvBNReLU and the rest get Transpose blocks
    of stride ``strides[i]``.

    For the production config (``in_channels=[64,128,256]``,
    ``out_channels=[128,128,128]``, ``strides=[1,2]``): ``len(strides)=2`` !=
    ``len(in_channels)=3``, so the first block is a ConvBNReLU stride-2 and
    the next two are Transpose(kernel=stride=strides[i]).
    """

    def __init__(self, in_channels, out_channels, strides):
        super(MultiMAPFusion, self).__init__()
        blocks = []
        if len(strides) == len(in_channels):
            assert len(strides) == len(out_channels), \
                'in_channels, out_channels, and strides should be in the same length for upsampling to the largest scale.'
            for i in range(len(in_channels)):
                blocks.append(
                    Transpose(
                        in_channels=in_channels[i],
                        out_channels=out_channels[i],
                        kernel_size=strides[i],
                        stride=strides[i])
                )
        else:
            blocks.append(
                ConvBNReLU(
                    in_channels=in_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=2
                )
            )

            for i in range(len(in_channels) - 1):
                blocks.append(
                    Transpose(
                        in_channels=in_channels[i + 1],
                        out_channels=out_channels[i + 1],
                        kernel_size=strides[i],
                        stride=strides[i])
                )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs):
        outs = []

        for i, x in enumerate(inputs):
            outs.append(self.blocks[i](x))

        return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------- #
# DeformLayer                                                                  #
# --------------------------------------------------------------------------- #
class DeformLayer(nn.Module):
    """A deformable-conv stack that wraps DCNv3_pytorch.

    Faithful port of ``common.DeformLayer``. When ``use_ffn=False`` (the
    production case) it holds a BARE ``DCNv3_pytorch`` (no FFN, no extra
    norm/act unless ``use_norm=True``). When ``channels`` is a single int the
    layer processes ONE feature map (the ``former_deform2`` case); when it is
    a list it processes multi-scale features.

    The DCNv3 input is channels-last; the layer permutes
    ``(N,C,H,W) -> (N,H,W,C)`` before DCNv3 and permutes back afterwards,
    matching the original.
    """

    def __init__(
        self,
        channels,
        group,
        offset_scale=2.0,
        use_ffn=False,
        use_norm=False,
    ):
        super().__init__()

        self.channels_list = isinstance(channels, list)

        if not self.channels_list:  # single feature map
            assert isinstance(channels, int), \
                'channels has to be a list of ints or an int'
            if not use_ffn:
                block = [
                    DCNv3_pytorch(channels=channels, group=group,
                                  offset_scale=offset_scale)
                ]
                if use_norm:
                    from pcdet.ops.dcnv3 import build_norm_layer, build_act_layer
                    block.append(build_norm_layer(channels, 'BN'))
                    block.append(build_act_layer('ReLU'))
            else:
                raise NotImplementedError(
                    'DeformFFN path is not ported (production config sets '
                    'use_ffn=False). Add it here if a future config needs it.')
            self.blocks = nn.Sequential(*block)
        else:  # multi-scale list of feature maps
            blocks = []
            for i, channel in enumerate(channels):
                if not use_ffn:
                    block = [
                        DCNv3_pytorch(channels=channel, group=group,
                                      offset_scale=offset_scale)
                    ]
                    if use_norm:
                        from pcdet.ops.dcnv3 import build_norm_layer, build_act_layer
                        block.append(build_norm_layer(channels, 'BN'))
                        block.append(build_act_layer('ReLU'))
                else:
                    raise NotImplementedError(
                        'DeformFFN path is not ported (production config sets '
                        'use_ffn=False).')
                block = nn.Sequential(*block)
                blocks.append(block)

            self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs):
        if self.channels_list:
            outs = []
            for i, x in enumerate(inputs):
                x = self.blocks[i](x.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
                outs.append(x)
            return tuple(outs)
        else:
            assert isinstance(inputs, torch.Tensor), \
                'the input has to be a single torch.Tensor for this single DCN block'
            out = self.blocks(inputs.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            return out


# --------------------------------------------------------------------------- #
# MDFENNeck                                                                    #
# --------------------------------------------------------------------------- #
class MDFENNeck(nn.Module):
    """Multi-scale Deformable PAN Neck (OpenPCDet port).

    Faithful port of ``MDFENNeck.MDFENNeck``. The PAN bidirectional flow is
    preserved verbatim:

      Top-down (FPN half):
        x0 -> reduce_layer0 -> fpn_out0 -> upsample0 -> cat with x1 -> Rep_p4
        f_out0 -> reduce_layer1 -> fpn_out1 -> upsample1 -> cat with x2 -> Rep_p3
        -> pan_out2
      Bottom-up (PAN half):
        pan_out2 -> downsample2 -> cat with fpn_out1 -> Rep_n3 -> pan_out1
        pan_out1 -> downsample1 -> cat with fpn_out0 -> Rep_n4 -> pan_out0

    With the production config (``dcn_layer=False, former=True, latter=False,
    dcn_ids=[2]``), the ONLY DCN is ``former_deform2`` applied to the
    concatenated feature ``f_concat_layer1`` (channels_list[0]+channels_list[4])
    right before ``Rep_p3``.

    ``forward`` accepts ``(x2, x1, x0)`` in the mmdet3d convention
    (smallest-spatial first) and returns either the 3-scale PAN output list
    or, when ``multi_fusion=True``, ``[fusion(outs)]`` (a one-element list
    whose only entry is the fused ``(B, sum(fused_channels), H, W)`` map).
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        dcn_layer=True,
        dcn_index=[0],
        dcn_ids=[2],
        former=True,
        latter=False,
        group=4,
        use_ffn=False,
        use_norm=False,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
        use_dwconv=True,
        use_normconv=False,
        multi_fusion=True,
        fused_channels=[128, 128, 128],
        fusion_strides=[1, 2],
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        assert not (former and latter), \
            'former and latter cant be True simultaneously.'
        self.former = former
        self.latter = latter

        # Define the positions of deformable convolutions
        self.dcn_layer = dcn_layer  # True -> before the PAN; False -> inside PAN
        self.dcn_index = dcn_index
        self.dcn_ids = dcn_ids

        if dcn_layer:
            # Independent pre-PAN DCN stack. NOT used in the production config
            # (dcn_layer=False), but preserved for fidelity.
            raise NotImplementedError(
                'dcn_layer=True path (FastDeformLayer pre-PAN) is not ported '
                '(production config uses dcn_layer=False).')

        if not dcn_layer and former and 0 in dcn_ids:
            self.former_deform0 = DeformLayer(
                channels=channels_list[2],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[2],
            out_channels=channels_list[3],
            kernel_size=1,
            stride=1
        )

        if not dcn_layer and latter and 0 in dcn_ids:
            self.latter_deform0 = DeformLayer(
                channels=channels_list[3],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.upsample0 = Transpose(
            in_channels=channels_list[3],
            out_channels=channels_list[3],
        )

        if not dcn_layer and former and 1 in dcn_ids:
            self.former_deform1 = DeformLayer(
                channels=channels_list[1] + channels_list[3],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[1] + channels_list[3],
            out_channels=channels_list[3],
            n=num_repeats[0],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[3],
            out_channels=channels_list[4],
            kernel_size=1,
            stride=1
        )

        if not dcn_layer and latter and 1 in dcn_ids:
            self.latter_deform1 = DeformLayer(
                channels=channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.upsample1 = Transpose(
            in_channels=channels_list[4],
            out_channels=channels_list[4]
        )

        if not dcn_layer and former and 2 in dcn_ids:
            # *** The production DCN site (dcn_ids=[2], former=True). ***
            # Input channel = channels_list[0] + channels_list[4] = 128.
            self.former_deform2 = DeformLayer(
                channels=channels_list[0] + channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[0] + channels_list[4],
            out_channels=channels_list[4],
            n=num_repeats[1],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        if not dcn_layer and latter and 2 in dcn_ids:
            self.latter_deform2 = DeformLayer(
                channels=channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[4],
            out_channels=channels_list[4],
            kernel_size=3,
            stride=2
        )

        if not dcn_layer and former and 3 in dcn_ids:
            self.former_deform3 = DeformLayer(
                channels=channels_list[4] + channels_list[4],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[4] + channels_list[4],
            out_channels=channels_list[5],
            n=num_repeats[2],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        if not dcn_layer and latter and 3 in dcn_ids:
            self.latter_deform3 = DeformLayer(
                channels=channels_list[5],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
            kernel_size=3,
            stride=2
        )

        if not dcn_layer and former and 4 in dcn_ids:
            self.former_deform4 = DeformLayer(
                channels=channels_list[3] + channels_list[5],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[6],
            n=num_repeats[3],
            inference_mode=inference_mode,
            use_se=use_se,
            num_conv_branches=num_conv_branches,
            use_dwconv=use_dwconv,
            use_normconv=use_normconv
        )

        if not dcn_layer and latter and 4 in dcn_ids:
            self.latter_deform4 = DeformLayer(
                channels=channels_list[6],
                group=group,
                use_ffn=use_ffn,
                use_norm=use_norm
            )

        self.multi_fusion = multi_fusion
        if self.multi_fusion:
            self.fusion = MultiMAPFusion(
                in_channels=channels_list[4:],
                out_channels=fused_channels,
                strides=fusion_strides
            )

    def forward(self, input):
        """Forward.

        Args:
            input (list or tuple of 3 tensors): the multi-scale features from
                the backbone, in mmdet3d order ``(x2, x1, x0)`` where ``x2``
                has the SMALLEST spatial size and ``x0`` the LARGEST.

        Returns:
            list[Tensor]: if ``multi_fusion`` is True, ``[fused]`` where
            ``fused`` has shape ``(B, sum(fused_channels), H_largest, W_largest)``;
            otherwise ``[pan_out2, pan_out1, pan_out0]``.
        """
        # dcn_layer is False (production); feed backbone features straight in.
        (x2, x1, x0) = input

        # --- top-down (FPN half) ---
        # branch 0: x0 -> [former_deform0] -> reduce -> [latter_deform0] -> fpn_out0
        if not self.dcn_layer and self.former and 0 in self.dcn_ids:
            dcn_out0 = self.former_deform0(x0)
        else:
            dcn_out0 = x0

        if not self.dcn_layer and self.latter and 0 in self.dcn_ids:
            fpn_out0 = self.latter_deform0(self.reduce_layer0(dcn_out0))
        else:
            fpn_out0 = self.reduce_layer0(dcn_out0)

        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)

        # branch 1: f_concat_layer0 -> [former_deform1] -> Rep_p4
        if not self.dcn_layer and self.former and 1 in self.dcn_ids:
            dcn_out1 = self.former_deform1(f_concat_layer0)
        else:
            dcn_out1 = f_concat_layer0

        f_out0 = self.Rep_p4(dcn_out1)

        if not self.dcn_layer and self.latter and 1 in self.dcn_ids:
            fpn_out1 = self.latter_deform1(self.reduce_layer1(f_out0))
        else:
            fpn_out1 = self.reduce_layer1(f_out0)

        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)

        # branch 2: f_concat_layer1 -> [former_deform2] -> Rep_p3 -> [latter_deform2] -> pan_out2
        if not self.dcn_layer and self.former and 2 in self.dcn_ids:
            dcn_out2 = self.former_deform2(f_concat_layer1)
        else:
            dcn_out2 = f_concat_layer1

        if not self.dcn_layer and self.latter and 2 in self.dcn_ids:
            pan_out2 = self.latter_deform2(self.Rep_p3(dcn_out2))
        else:
            pan_out2 = self.Rep_p3(dcn_out2)

        # --- bottom-up (PAN half) ---
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)

        # branch 3
        if not self.dcn_layer and self.former and 3 in self.dcn_ids:
            dcn_out3 = self.former_deform3(p_concat_layer1)
        else:
            dcn_out3 = p_concat_layer1

        if not self.dcn_layer and self.latter and 3 in self.dcn_ids:
            pan_out1 = self.latter_deform3(self.Rep_n3(dcn_out3))
        else:
            pan_out1 = self.Rep_n3(dcn_out3)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)

        # branch 4
        if not self.dcn_layer and self.former and 4 in self.dcn_ids:
            dcn_out4 = self.former_deform4(p_concat_layer2)
        else:
            dcn_out4 = p_concat_layer2

        if not self.dcn_layer and self.latter and 4 in self.dcn_ids:
            pan_out0 = self.latter_deform4(self.Rep_n4(dcn_out4))
        else:
            pan_out0 = self.Rep_n4(dcn_out4)

        outputs = [pan_out2, pan_out1, pan_out0]

        if self.multi_fusion:
            return [self.fusion(outputs)]
        else:
            return outputs
