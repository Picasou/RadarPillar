"""Shared FPN-variant model configs (port + original sides).

Mirrors the values in
``projects/RadarNeXt/configs/radarnext_fpn_variant.py`` and the matching
OpenPCDet port (audit M4: NUM_BEV_FEATURES=32 instead of the original 64).
"""

import copy

from easydict import EasyDict

# Geometry (VoD).
VOXEL_SIZE = [0.16, 0.16, 5]
POINT_CLOUD_RANGE = [0, -25.6, -3, 51.2, 25.6, 2]
GRID_SIZE = [320, 320, 1]
CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']  # original ordering
TASKS = [dict(num_class=3, class_names=['Pedestrian', 'Cyclist', 'Car'])]

# Port-side (OpenPCDet) class-name ordering (audit-correct).
CLASS_NAMES_PORT = ['Car', 'Pedestrian', 'Cyclist']

# RepDWC hyperparameters (identical on both sides).
REPDWC_IN_CHANNELS = 32   # M4 audit: NUM_BEV_FEATURES=32, NOT the mmdet3d 64.
REPDWC_OUT_CHANNELS = [64, 128, 256]
REPDWC_LAYER_NUMS = [3, 5, 5]
REPDWC_LAYER_STRIDES = [2, 2, 2]
REPDWC_NUM_OUTPUTS = 3

# SECONDFPN hyperparameters (identical on both sides).
FPN_IN_CHANNELS = [64, 128, 256]
FPN_OUT_CHANNELS = [128, 128, 128]
FPN_UPSAMPLE_STRIDES = [0.5, 1, 2]
FPN_USE_CONV_FOR_NO_STRIDE = True
FPN_NORM_CFG = dict(type='BN', eps=1e-3, momentum=0.01)
FPN_UPSAMPLE_CFG = dict(type='deconv', bias=False)
FPN_CONV_CFG = dict(type='Conv2d', bias=False)

# CenterHead hyperparameters (identical on both sides).
HEAD_IN_CHANNELS = sum(FPN_OUT_CHANNELS)  # 384
HEAD_STRIDES = [2]  # FPN variant: 80 -> 160
HEAD_WEIGHT = 1.0
HEAD_CORNER_WEIGHT = 1.0
HEAD_IOU_WEIGHT = 1.0
HEAD_IOU_REG_WEIGHT = 0.5
HEAD_RECTIFIER = [[0.5, 0.5, 0.5]]
HEAD_CODE_WEIGHTS = [1.0] * 8
HEAD_COMMON_HEADS = {
    'reg': (2, 2),
    'height': (1, 2),
    'dim': (3, 2),
    'rot': (2, 2),
    'iou': (1, 2),
}
HEAD_WITH_CORNER = True
HEAD_WITH_REG_IOU = True
HEAD_SHARE_CONV_CHANNEL = 64
HEAD_NUM_HM_CONV = 2
HEAD_NUM_CORNER_HM_CONV = 2
HEAD_INIT_BIAS = -2.19
HEAD_FINAL_KERNEL = 3
HEAD_OUT_SIZE_FACTOR = 2
HEAD_BBOX_CODE_SIZE = 7
HEAD_MAX_OBJS = 500
HEAD_DENSE_REG = 1
HEAD_GAUSSIAN_OVERLAP = 0.1
HEAD_MIN_RADIUS = 2


# --------------------------------------------------------------------------- #
# Port (OpenPCDet) cfg builders                                                #
# --------------------------------------------------------------------------- #
def build_repdwc_cfg_port():
    return EasyDict({
        'OUT_CHANNELS': list(REPDWC_OUT_CHANNELS),
        'LAYER_NUMS': list(REPDWC_LAYER_NUMS),
        'LAYER_STRIDES': list(REPDWC_LAYER_STRIDES),
        'NUM_OUTPUTS': REPDWC_NUM_OUTPUTS,
        'INFERENCE_MODE': False,
        'USE_SE': False,
        'NUM_CONV_BRANCHES': 1,
        'USE_NORMCONV': False,
        'USE_DWCONV': True,
    })


def build_secondfpn_cfg_port():
    return EasyDict({
        'IN_CHANNELS': list(FPN_IN_CHANNELS),
        'OUT_CHANNELS': list(FPN_OUT_CHANNELS),
        'UPSAMPLE_STRIDES': list(FPN_UPSAMPLE_STRIDES),
        'USE_CONV_FOR_NO_STRIDE': FPN_USE_CONV_FOR_NO_STRIDE,
        'NORM_CFG': dict(FPN_NORM_CFG),
        'UPSAMPLE_CFG': dict(FPN_UPSAMPLE_CFG),
        'CONV_CFG': dict(FPN_CONV_CFG),
    })


def build_backbone_fpn_cfg_port():
    return EasyDict({
        'NAME': 'RadarNeXtFPNBackbone',
        'REP_DWC': build_repdwc_cfg_port(),
        'SECOND_FPN': build_secondfpn_cfg_port(),
    })


def build_head_cfg_port():
    return EasyDict({
        'NAME': 'RadarNeXtCenterHead',
        'TASKS': [EasyDict(t) for t in TASKS],
        'CODE_WEIGHTS': list(HEAD_CODE_WEIGHTS),
        'WEIGHT': HEAD_WEIGHT,
        'CORNER_WEIGHT': HEAD_CORNER_WEIGHT,
        'IOU_WEIGHT': HEAD_IOU_WEIGHT,
        'IOU_REG_WEIGHT': HEAD_IOU_REG_WEIGHT,
        'STRIDES': list(HEAD_STRIDES),
        'RECTIFIER': copy.deepcopy(HEAD_RECTIFIER),
        'BBOX_CODE_SIZE': HEAD_BBOX_CODE_SIZE,
        'COMMON_HEADS': copy.deepcopy(HEAD_COMMON_HEADS),
        'WITH_CORNER': HEAD_WITH_CORNER,
        'WITH_REG_IOU': HEAD_WITH_REG_IOU,
        'SHARE_CONV_CHANNEL': HEAD_SHARE_CONV_CHANNEL,
        'NUM_HM_CONV': HEAD_NUM_HM_CONV,
        'NUM_CORNER_HM_CONV': HEAD_NUM_CORNER_HM_CONV,
        'INIT_BIAS': HEAD_INIT_BIAS,
        'FINAL_KERNEL': HEAD_FINAL_KERNEL,
        'OUT_SIZE_FACTOR': HEAD_OUT_SIZE_FACTOR,
        'MAX_OBJS': HEAD_MAX_OBJS,
        'DENSE_REG': HEAD_DENSE_REG,
        'GAUSSIAN_OVERLAP': HEAD_GAUSSIAN_OVERLAP,
        'MIN_RADIUS': HEAD_MIN_RADIUS,
        'POST_CENTER_LIMIT_RANGE': list(POINT_CLOUD_RANGE),
        'SCORE_THRESHOLD': 0.1,
        'NMS_CONFIG': EasyDict(
            NMS_THRESH=0.2, NMS_PRE_MAXSIZE=1000, NMS_POST_MAXSIZE=83),
    })


# --------------------------------------------------------------------------- #
# Original (mmdet3d) cfg builders (plain dicts; originals take kwargs anyway)  #
# --------------------------------------------------------------------------- #
def build_repdwc_kwargs_orig():
    return dict(
        in_channels=REPDWC_IN_CHANNELS,
        out_channels=list(REPDWC_OUT_CHANNELS),
        layer_nums=list(REPDWC_LAYER_NUMS),
        layer_strides=list(REPDWC_LAYER_STRIDES),
        num_outputs=REPDWC_NUM_OUTPUTS,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
        use_normconv=False,
        use_dwconv=True,
    )


def build_secondfpn_kwargs_orig():
    return dict(
        in_channels=list(FPN_IN_CHANNELS),
        out_channels=list(FPN_OUT_CHANNELS),
        upsample_strides=list(FPN_UPSAMPLE_STRIDES),
        norm_cfg=dict(FPN_NORM_CFG),
        upsample_cfg=dict(FPN_UPSAMPLE_CFG),
        conv_cfg=dict(FPN_CONV_CFG),
        use_conv_for_no_stride=FPN_USE_CONV_FOR_NO_STRIDE,
    )


def build_head_kwargs_orig():
    """Original RadarNeXt_Head constructor kwargs (FPN variant)."""
    return dict(
        in_channels=HEAD_IN_CHANNELS,
        multi_fusion=False,
        fusion_channels=list(FPN_OUT_CHANNELS),
        fusion_strides=[1, 2],
        tasks=[dict(t) for t in TASKS],
        strides=list(HEAD_STRIDES),
        weight=HEAD_WEIGHT,
        corner_weight=HEAD_CORNER_WEIGHT,
        iou_weight=HEAD_IOU_WEIGHT,
        iou_reg_weight=HEAD_IOU_REG_WEIGHT,
        code_weights=list(HEAD_CODE_WEIGHTS),
        common_heads=copy.deepcopy(HEAD_COMMON_HEADS),
        with_corner=HEAD_WITH_CORNER,
        with_reg_iou=HEAD_WITH_REG_IOU,
        voxel_size=list(VOXEL_SIZE),
        pc_range=list(POINT_CLOUD_RANGE),
        out_size_factor=HEAD_OUT_SIZE_FACTOR,
        rectifier=copy.deepcopy(HEAD_RECTIFIER),
        bbox_code_size=HEAD_BBOX_CODE_SIZE,
        init_bias=HEAD_INIT_BIAS,
        share_conv_channel=HEAD_SHARE_CONV_CHANNEL,
        num_hm_conv=HEAD_NUM_HM_CONV,
        num_corner_hm_conv=HEAD_NUM_CORNER_HM_CONV,
        train_cfg=dict(
            grid_size=GRID_SIZE[:2],
            voxel_size=list(VOXEL_SIZE),
            out_size_factor=HEAD_OUT_SIZE_FACTOR,
            dense_reg=HEAD_DENSE_REG,
            gaussian_overlap=HEAD_GAUSSIAN_OVERLAP,
            point_cloud_range=list(POINT_CLOUD_RANGE),
            max_objs=HEAD_MAX_OBJS,
            min_radius=HEAD_MIN_RADIUS,
            code_weights=list(HEAD_CODE_WEIGHTS),
        ),
    )
