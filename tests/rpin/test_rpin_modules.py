"""RPiN 7 模块契约 + PP* 4-bug + head_2d 修复回归测试（DoD2）。"""
import os
import sys

import numpy as np
import pytest
import torch

from tests.rpin import _load_model_cfg


# ---------------------------------------------------------------------------
# A2 SEBlock / A3 SEDWConv：BACKBONE_3D 契约
# ---------------------------------------------------------------------------
def test_seblock_contract():
    from pcdet.models.backbones_3d.se_block import SEBlock
    mcfg = _load_model_cfg('a2').BACKBONE_3D
    m = SEBlock(mcfg, input_channels=32, grid_size=[320, 320, 1])
    assert m.num_point_features == 32
    bd = {'pillar_features': torch.randn(200, 32)}
    out = m(bd)
    assert out['pillar_features'].shape == (200, 32)   # 就地重标定，shape 不变


def test_sedwconv_contract():
    from pcdet.models.backbones_3d.se_dwconv import SEDWConv
    mcfg = _load_model_cfg('a3').BACKBONE_3D
    m = SEDWConv(mcfg, input_channels=32, grid_size=[320, 320, 1])
    assert m.num_point_features == 32
    M = 200
    coords = torch.stack([
        torch.zeros(M), torch.zeros(M),
        torch.randint(0, 320, (M,)).float(), torch.randint(0, 320, (M,)).float(),
    ], dim=1)
    bd = {'pillar_features': torch.randn(M, 32), 'voxel_coords': coords}
    out = m(bd)
    assert out['pillar_features'].shape[0] == M


# ---------------------------------------------------------------------------
# PPFPN / PPMDFEN：4-bug 修复回归
# ---------------------------------------------------------------------------
def test_ppfpn_4bug():
    """PP* 4-bug：① 取 fpn()[0]（tensor 非列表）② 无 input_channels 参 ③ 大→小 ④ num_bev=sum(OUT)。"""
    from pcdet.models.backbones_2d.pp_fpn import PPFPNBackbone
    mcfg = _load_model_cfg('n2').BACKBONE_2D
    in_ch = int(_load_model_cfg('n2').MAP_TO_BEV.NUM_BEV_FEATURES)
    m = PPFPNBackbone(mcfg, input_channels=in_ch)
    # ④ num_bev_features == sum(SECOND_FPN.OUT_CHANNELS)
    assert m.num_bev_features == sum(mcfg.SECOND_FPN.OUT_CHANNELS)
    bd = {'spatial_features': torch.randn(1, in_ch, 320, 320)}
    sf2d = m(bd)['spatial_features_2d']
    assert torch.is_tensor(sf2d)                       # ① 取 [0] 得 tensor（非 list）
    assert sf2d.shape[1] == m.num_bev_features          # 通道 = num_bev
    assert sf2d.shape[2] == 160                         # FPN 融合到 160×160


def test_ppmdfen_fused_shape():
    from pcdet.models.backbones_2d.pp_mdfen import PPMDFENBackbone
    mcfg = _load_model_cfg('n3').BACKBONE_2D
    in_ch = int(_load_model_cfg('n3').MAP_TO_BEV.NUM_BEV_FEATURES)
    m = PPMDFENBackbone(mcfg, input_channels=in_ch)
    assert m.num_bev_features == 384                    # sum(FUSED_CHANNELS=[128]*3)
    bd = {'spatial_features': torch.randn(1, in_ch, 320, 320)}
    sf2d = m(bd)['spatial_features_2d']
    assert tuple(sf2d.shape[1:]) == (384, 80, 80)       # MDFEN 融合到 80×80（忠实原版）


def test_repdwcnone_outs0_design():
    """n4 设计裁决：取 outs[0]=160×160（与 n1 同分辨率，隔离块类型消融变量）。"""
    from pcdet.models.backbones_2d.repdwc_none import RepDWCNoneBackbone
    mcfg = _load_model_cfg('n4').BACKBONE_2D
    in_ch = int(_load_model_cfg('n4').MAP_TO_BEV.NUM_BEV_FEATURES)
    m = RepDWCNoneBackbone(mcfg, input_channels=in_ch)
    assert m.num_bev_features == list(mcfg.OUT_CHANNELS)[0] == 64
    bd = {'spatial_features': torch.randn(1, in_ch, 320, 320)}
    sf2d = m(bd)['spatial_features_2d']
    assert tuple(sf2d.shape[1:]) == (64, 160, 160)      # 首层 64ch@160×160


# ---------------------------------------------------------------------------
# RadarNeXtCenterHead2D：继承 + override（H2 不再 override forward）+ H3 按类填 z
# ---------------------------------------------------------------------------
def test_head2d_inheritance_and_per_class_height():
    from pcdet.models.dense_heads.radarnext_center_head_2d import RadarNeXtCenterHead2D
    from pcdet.models.dense_heads.radarnext_center_head import RadarNeXtCenterHead
    assert issubclass(RadarNeXtCenterHead2D, RadarNeXtCenterHead)
    # H2：override predict/_override_height，不 override forward（避免 eval 崩）
    assert 'predict' in RadarNeXtCenterHead2D.__dict__
    assert '_override_height' in RadarNeXtCenterHead2D.__dict__
    assert 'forward' not in RadarNeXtCenterHead2D.__dict__


def test_head2d_per_class_height_fill():
    """H3：_override_height 按 cell 预测类别填各类 anchor 底高（非跨类均值）。"""
    from pcdet.models.dense_heads.radarnext_center_head_2d import RadarNeXtCenterHead2D
    from pcdet.config import cfg_from_yaml_file
    from pcdet.models import build_network
    local_cfg = EasyDict() if False else None  # placeholder; build below
    import os
    from easydict import EasyDict as _ED
    lc = _ED()
    cfg_from_yaml_file(os.path.join(_ROOT, 'experiments', 'YAML', 'head_2d.yaml'), lc)
    pcr = lc.DATA_CONFIG.POINT_CLOUD_RANGE
    vs = np.array(lc.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE'])
    gs = np.array([int((pcr[3]-pcr[0])/vs[0]), int((pcr[4]-pcr[1])/vs[1]), int((pcr[5]-pcr[2])/vs[2])], np.int32)

    class DS:
        pass
    ds = DS()
    ds.class_names = list(lc.CLASS_NAMES)
    ds.point_feature_encoder = DS()
    ds.point_feature_encoder.num_point_features = 9
    ds.grid_size = gs
    ds.voxel_size = vs
    ds.point_cloud_range = list(pcr)
    model = build_network(model_cfg=lc.MODEL, num_class=len(ds.class_names), dataset=ds)
    head = model.dense_head
    assert isinstance(head, RadarNeXtCenterHead2D)

    # 构造 3 类各一点的 hm：class0@(0,0) class1@(1,1) class2@(2,2)
    hm = torch.zeros(1, 3, 4, 4)
    hm[0, 0, 0, 0] = 5
    hm[0, 1, 1, 1] = 5
    hm[0, 2, 2, 2] = 5
    pd = {'hm': hm, 'height': torch.zeros(1, 1, 4, 4)}
    head._override_height([pd])
    z = pd['height'][0, 0]
    anchors = head.anchor_bottom_heights
    assert abs(z[0, 0].item() - anchors[0]) < 1e-4   # class0
    assert abs(z[1, 1].item() - anchors[1]) < 1e-4   # class1
    assert abs(z[2, 2].item() - anchors[2]) < 1e-4   # class2
    # 非跨类均值
    mean_val = sum(anchors) / len(anchors)
    assert abs(z[0, 0].item() - mean_val) > 0.1


# ---------------------------------------------------------------------------
# E3 VDC：纯函数 + 常量 + time_scale 量纲
# ---------------------------------------------------------------------------
def test_vdc_contract():
    from pcdet.datasets.vod import vdc
    assert hasattr(vdc, 'compensate_motion')
    assert callable(vdc.compensate_motion)
    assert vdc.RADAR_FEATURE_ORDER == ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
    vod_ds_src = open(os.path.join(_ROOT, 'pcdet', 'datasets', 'vod', 'vod_dataset.py')).read()
    assert 'USE_VDC' in vod_ds_src


def test_vdc_compensate_motion_is_pure_and_scaled():
    """compensate_motion 不改输入；time_scale 控制位移量纲（帧索引需 ×0.1 才为秒）。"""
    from pcdet.datasets.vod.vdc import compensate_motion
    # (x=10, y=0, v_r=5, time=-2帧)
    pts = np.array([[10.0, 0.0, 0.0, 1.0, 5.0, 5.0, -2.0]])
    orig = pts.copy()
    out = compensate_motion(pts, cfg={'time_scale': 0.1, 'use_vr_comp': True})
    assert np.array_equal(pts, orig)                    # 纯函数：不改输入
    # x 方向、v_r=5、t=-2*0.1=-0.2s → dx = v*t = 5*(-0.2) = -1.0 → x_new = 10 - (-1.0) = 11.0
    assert abs(out[0, 0] - 11.0) < 1e-3
    # time_scale=1.0（误用）→ 补偿 10×（11 → 20）
    out_wrong = compensate_motion(pts, cfg={'time_scale': 1.0})
    assert abs(out_wrong[0, 0] - 20.0) < 1e-3


# 提供 _ROOT（test_vdc_contract 用）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from easydict import EasyDict  # noqa: E402  (late import to keep file head clean)
