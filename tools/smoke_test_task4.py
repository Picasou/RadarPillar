"""Task 4 smoke test: build CenterPoint (RadarNeXtCenterHead) end-to-end,
feed a fake batch_dict, run forward -> get_loss, assert finite positive
scalar with all sub-loss keys.

Uses FPN-variant model_cfg values (the head is parametrized; Task 5/7 will
write the actual YAML). Runs on CPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from easydict import EasyDict

from pcdet.models.dense_heads.radarnext_center_head import RadarNeXtCenterHead
from pcdet.models.detectors.centerpoint import CenterPoint


def build_head_model_cfg():
    """FPN-variant head config (radarnext_fpn_variant.py values)."""
    cfg = EasyDict()
    cfg.NAME = 'RadarNeXtCenterHead'
    cfg.TASKS = [
        EasyDict(num_class=3, class_names=['Car', 'Pedestrian', 'Cyclist']),
    ]
    cfg.CODE_WEIGHTS = [1.0] * 8
    cfg.WEIGHT = 1.0
    cfg.CORNER_WEIGHT = 1.0
    cfg.IOU_WEIGHT = 1.0
    cfg.IOU_REG_WEIGHT = 0.5
    cfg.STRIDES = [2]  # FPN variant: 80 -> 160
    cfg.RECTIFIER = [[0.5, 0.5, 0.5]]
    cfg.BBOX_CODE_SIZE = 7
    cfg.COMMON_HEADS = {
        'reg': (2, 2),
        'height': (1, 2),
        'dim': (3, 2),
        'rot': (2, 2),
        'iou': (1, 2),
    }
    cfg.WITH_CORNER = True
    cfg.WITH_REG_IOU = True
    cfg.SHARE_CONV_CHANNEL = 64
    cfg.NUM_HM_CONV = 2
    cfg.NUM_CORNER_HM_CONV = 2
    cfg.INIT_BIAS = -2.19
    cfg.FINAL_KERNEL = 3
    cfg.OUT_SIZE_FACTOR = 2
    cfg.MAX_OBJS = 500
    cfg.DENSE_REG = 1
    cfg.GAUSSIAN_OVERLAP = 0.1
    cfg.MIN_RADIUS = 2
    # Eval-side
    cfg.POST_CENTER_LIMIT_RANGE = [0, -25.6, -3, 51.2, 25.6, 2]
    cfg.SCORE_THRESHOLD = 0.1
    cfg.NMS_CONFIG = EasyDict(
        NMS_THRESH=0.2, NMS_PRE_MAXSIZE=1000, NMS_POST_MAXSIZE=83,
    )
    return cfg


def make_fake_batch(B=2, M=8, device='cuda'):
    """Fake batch_dict with gt_boxes (B, M, 8) and spatial_features_2d."""
    torch.manual_seed(0)
    pc_range = [0, -25.6, -3, 51.2, 25.6, 2]
    # Random gt boxes inside the point cloud range.
    cx = torch.empty(B, M).uniform_(5, 45)
    cy = torch.empty(B, M).uniform_(-20, 20)
    cz = torch.empty(B, M).uniform_(-2, 1)
    dx = torch.empty(B, M).uniform_(1.5, 4.0)
    dy = torch.empty(B, M).uniform_(0.5, 1.8)
    dz = torch.empty(B, M).uniform_(1.2, 1.8)
    rot = torch.empty(B, M).uniform_(-3.14, 3.14)
    cls = torch.randint(1, 4, (B, M)).float()  # 1-based 1..3
    gt_boxes = torch.stack([cx, cy, cz, dx, dy, dz, rot, cls], dim=-1).to(device)
    spatial_features_2d = torch.randn(B, 384, 80, 80, device=device)
    return {
        'batch_size': B,
        'gt_boxes': gt_boxes,
        'spatial_features_2d': spatial_features_2d,
    }


def test_head_only():
    print('=== Head-only forward + get_loss ===')
    head_cfg = build_head_model_cfg()
    grid_size = [320, 320, 1]
    pc_range = [0, -25.6, -3, 51.2, 25.6, 2]
    head = RadarNeXtCenterHead(
        model_cfg=head_cfg, input_channels=384, num_class=3,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        grid_size=grid_size, point_cloud_range=pc_range,
        predict_boxes_when_training=True,
    )
    head.train()
    head.cuda()
    batch = make_fake_batch()
    data_dict = head(batch)
    assert head.forward_ret_dict is not None, 'forward_ret_dict not set in training'
    loss, tb_dict = head.get_loss()
    print('  loss =', float(loss))
    print('  tb_dict keys =', sorted(tb_dict.keys()))
    assert torch.isfinite(loss), 'loss is not finite'
    assert float(loss) > 0, 'loss is not positive'
    # All sub-loss keys present (task_id=0).
    expected = {'0_hm_loss', '0_loc_loss', '0_corner_loss', '0_iou_loss', '0_iou_reg_loss'}
    missing = expected - set(tb_dict.keys())
    assert not missing, f'missing sub-loss keys: {missing}'
    print('  HEAD-ONLY: PASS')
    return head


def test_head_eval():
    print('=== Head eval forward (predict/post_processing) ===')
    head_cfg = build_head_model_cfg()
    head = RadarNeXtCenterHead(
        model_cfg=head_cfg, input_channels=384, num_class=3,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        grid_size=[320, 320, 1], point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        predict_boxes_when_training=True,
    )
    head.eval()
    head.cuda()
    batch = make_fake_batch(B=1)
    with torch.no_grad():
        data_dict = head(batch)
    pred_dicts = data_dict['pred_dicts']
    assert len(pred_dicts) == 1, f'expected 1 per-sample dict, got {len(pred_dicts)}'
    pd = pred_dicts[0]
    print('  pred_boxes shape =', tuple(pd['pred_boxes'].shape))
    print('  pred_scores shape =', tuple(pd['pred_scores'].shape))
    print('  pred_labels shape =', tuple(pd['pred_labels'].shape))
    assert pd['pred_boxes'].shape[1] == 7, 'boxes should be 7-dim'
    # Labels are 1-based in OpenPCDet convention.
    if pd['pred_labels'].numel() > 0:
        assert pd['pred_labels'].min() >= 1 and pd['pred_labels'].max() <= 3, \
            f'labels out of [1,3]: min={pd["pred_labels"].min()} max={pd["pred_labels"].max()}'
    print('  HEAD-EVAL: PASS')


def test_detector():
    print('=== Detector end-to-end forward (train + eval) ===')
    head_cfg = build_head_model_cfg()

    # Minimal dataset shim: only the attributes build_networks reads.
    class _Shim:
        def __init__(self):
            self.class_names = ['Car', 'Pedestrian', 'Cyclist']
            self.grid_size = [320, 320, 1]
            self.point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
            self.voxel_size = [0.16, 0.16, 5]

            class _Enc:
                num_point_features = 7
            self.point_feature_encoder = _Enc()

    model_cfg = EasyDict()
    model_cfg.NAME = 'CenterPoint'
    model_cfg.VFE = EasyDict(NAME='DummyVFE')  # not used; we feed spatial_features_2d directly
    model_cfg.MAP_TO_BEV = EasyDict(NAME='DummyMap')  # not used
    model_cfg.BACKBONE_2D = EasyDict(NAME='DummyBackbone')  # not used
    model_cfg.DENSE_HEAD = head_cfg
    model_cfg.DENSE_HEAD.CLASS_AGNOSTIC = False
    model_cfg.DENSE_HEAD.NAME = 'RadarNeXtCenterHead'

    # We bypass the real module_topology by stubbing it: build only the head.
    det = CenterPoint.__new__(CenterPoint)
    torch.nn.Module.__init__(det)
    det.model_cfg = model_cfg
    det.num_class = 3
    det.dataset = _Shim()
    det.class_names = det.dataset.class_names
    det.register_buffer('global_step', torch.LongTensor(1).zero_())
    det.dense_head = RadarNeXtCenterHead(
        model_cfg=head_cfg, input_channels=384, num_class=3,
        class_names=det.class_names,
        grid_size=det.dataset.grid_size,
        point_cloud_range=det.dataset.point_cloud_range,
        predict_boxes_when_training=False,
    )
    det.module_list = [det.dense_head]
    det.init_weights()

    # Train forward.
    det.train()
    det.cuda()
    batch = make_fake_batch()
    ret_dict, tb_dict, disp_dict = det(batch)
    loss = ret_dict['loss']
    print('  train loss =', float(loss))
    assert torch.isfinite(loss) and float(loss) > 0, 'train loss invalid'
    print('  DETECTOR-TRAIN: PASS')

    # Eval forward.
    det.eval()
    batch_eval = make_fake_batch(B=1)
    batch_eval.pop('gt_boxes', None)
    with torch.no_grad():
        pred_dicts, recall = det(batch_eval)
    assert len(pred_dicts) == 1
    print('  eval #dets =', pred_dicts[0]['pred_boxes'].shape[0])
    print('  DETECTOR-EVAL: PASS')


if __name__ == '__main__':
    head = test_head_only()
    test_head_eval()
    test_detector()
    print('\nALL SMOKE TESTS PASSED')
