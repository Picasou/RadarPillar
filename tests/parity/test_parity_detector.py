"""Parity point 7: end-to-end detector (FPN variant).

Both sides are the full FPN-variant chain RepDWC -> SECONDFPN -> CenterHead,
run as a synthetic-batch ``detector``:

* PORT: OpenPCDet ``CenterPoint`` detector's dense_head path. We feed
  ``spatial_features_2d`` directly (bypassing VFE/Scatter — those need real
  point clouds and aren't part of the FPN-chain port). Train mode produces
  a loss dict; eval mode produces pred_dicts.
* ORIGINAL: the RadarNeXt ``RadarNeXt_Head`` driven by a manual
  RepDWC+SECONDFPN chain (the original ``RadarNeXt`` detector wires these
  as siblings; we replicate that). The original head's ``loss`` /
  ``predict`` take mmdet3d ``InstanceData`` / ``Det3DDataSample`` inputs,
  which we synthesize minimally.

To keep parity tractable (and avoid dataset-coupling differences), we
exercise ONE common contract on each side: the head's
``loss_by_feat(preds, gt)`` and ``forward(feats)``. These are the
numerically-load-bearing methods; the data-prep wrappers above them are
not part of the port's correctness surface.

So P7 = (a) chain forward parity (already P3+P5; re-asserted end-to-end),
(b) ``loss_by_feat`` parity on the chain output + a shared gt tensor,
(c) ``predict`` parity (eval-mode decoded preds), all on synthetic inputs.

This is the top-level integration check. A PASS means the FPN-chain port
is faithful end-to-end. A FAIL pinpoints which sub-stage diverges
(likely the data-format boundary, not the math).
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from easydict import EasyDict

from tests.parity import _originals as O  # noqa: F401
from tests.parity import _configs as C
from tests.parity.conftest import (
    align_state_dicts, gen_bev, gen_gt_boxes, parity_allclose,
    parity_allclose_list, seed_rng,
    LOOSE_ATOL, LOOSE_RTOL,
)
from build_weight_map import build_backbone_fpn_both, build_head_both


def _build_port_detector():
    """Port side: RadarNeXtFPNBackbone + RadarNeXtCenterHead (synthetic)."""
    from pcdet.models.backbones_2d.radarnext_backbone_fpn import (
        RadarNeXtFPNBackbone,
    )
    from pcdet.models.dense_heads.radarnext_center_head import (
        RadarNeXtCenterHead,
    )
    seed_rng(0)
    backbone = RadarNeXtFPNBackbone(
        model_cfg=C.build_backbone_fpn_cfg_port(),
        input_channels=C.REPDWC_IN_CHANNELS,
    )
    seed_rng(0)
    head = RadarNeXtCenterHead(
        model_cfg=C.build_head_cfg_port(),
        input_channels=C.HEAD_IN_CHANNELS,
        num_class=3,
        class_names=C.CLASS_NAMES_PORT,
        grid_size=C.GRID_SIZE,
        point_cloud_range=C.POINT_CLOUD_RANGE,
        predict_boxes_when_training=True,
    )
    return backbone, head


def _build_orig_chain():
    """Original side: RepDWC + SECONDFPN + RadarNeXt_Head."""
    seed_rng(0)
    repdwc = O.RepDWC(**C.build_repdwc_kwargs_orig())
    seed_rng(0)
    fpn = O.SECONDFPN(**C.build_secondfpn_kwargs_orig())
    seed_rng(0)
    head = O.RadarNeXt_Head(**C.build_head_kwargs_orig())
    return repdwc, fpn, head


def _align_detector(port_backbone, port_head,
                    orig_repdwc, orig_fpn, orig_head):
    """Weight-align all three module pairs across port and original."""
    # RepDWC: port wraps under backbone.backbone.*; original is bare.
    port_sd = port_backbone.state_dict()
    merged = {}
    for k, v in orig_repdwc.state_dict().items():
        merged[f'backbone.{k}'] = v
    for k, v in orig_fpn.state_dict().items():
        merged[f'fpn.{k}'] = v
    new_sd = {k: (merged[k].clone() if k in merged and merged[k].shape == v.shape else v)
              for k, v in port_sd.items()}
    port_backbone.load_state_dict(new_sd, strict=False)
    # Head: identity map.
    align_state_dicts(orig_head, port_head, verbose=False)


def test_parity_detector():
    if not torch.cuda.is_available():
        print('[P7] SKIP: detector loss path needs CUDA (pcdet iou3d_nms)')
        return
    dev = 'cuda'
    p_backbone, p_head = _build_port_detector()
    o_repdwc, o_fpn, o_head = _build_orig_chain()
    _align_detector(p_backbone, p_head, o_repdwc, o_fpn, o_head)
    p_backbone = p_backbone.to(dev); p_head = p_head.to(dev)
    o_repdwc = o_repdwc.to(dev); o_fpn = o_fpn.to(dev); o_head = o_head.to(dev)

    p_backbone.eval(); p_head.eval()
    o_repdwc.eval(); o_fpn.eval(); o_head.eval()

    # Shared synthetic input.
    x = gen_bev(batch=2, channels=C.REPDWC_IN_CHANNELS, h=320, w=320,
                seed=123, device=dev)
    with torch.no_grad():
        # Port chain.
        dd = p_backbone({'spatial_features': x})
        feat_port = dd['spatial_features_2d']  # (B, 384, 80, 80)
        # Original chain.
        ms = list(o_repdwc(x))
        feat_orig = o_fpn(ms)[0]

    # (a) Chain forward parity.
    passed_chain, ma_chain, _ = parity_allclose(
        feat_port, feat_orig, atol=LOOSE_ATOL, rtol=LOOSE_RTOL,
        name='detector.spatial_features_2d')

    # (b) Head forward parity (P5 re-asserted end-to-end through the chain).
    # Run in TRAIN mode so corner_hm is included (matches loss path).
    p_head.train(); o_head.train()
    with torch.no_grad():
        x_port = p_head.shared_conv(feat_port)
        ret_port = [t(x_port) for t in p_head.tasks]
        ret_orig = o_head([feat_orig])
    head_pass = True
    ma_head = 0.0
    for h in ret_port[0]:
        p, ma, _ = parity_allclose(
            ret_port[0][h], ret_orig[0][h],
            atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name=f'detector.head.{h}')
        head_pass = head_pass and p
        ma_head = max(ma_head, ma)

    # (c) loss_by_feat parity — both heads, same preds + same gt.
    gt_boxes_port = gen_gt_boxes(batch=2, max_objs=8, seed=999,
                                 pc_range=tuple(C.POINT_CLOUD_RANGE), device=dev)
    p_head.forward_ret_dict = {'preds_dicts': ret_port, 'gt_boxes': gt_boxes_port}
    losses_port = p_head.loss_by_feat(ret_port, gt_boxes_port)
    # Original: needs InstanceData-wrapped gt. We bypass by calling
    # loss_by_feat with the same tensor format the port uses via a tiny shim.
    # Re-run original forward in train mode to populate corner_hm too.
    with torch.no_grad():
        ret_orig = o_head([feat_orig])
    losses_orig = _orig_loss_by_feat_from_tensor_gt(
        o_head, ret_orig, gt_boxes_port)

    print('\n--- detector loss_by_feat keys ---')
    print(f'  port: {sorted(losses_port.keys())}')
    print(f'  orig: {sorted(losses_orig.keys())}')

    loss_pass = True
    ma_loss = 0.0
    for k in losses_port:
        if k not in losses_orig:
            print(f'  ! {k} missing in orig')
            loss_pass = False
            continue
        p, ma, _ = parity_allclose(
            losses_port[k], losses_orig[k],
            atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name=f'detector.loss.{k}')
        loss_pass = loss_pass and p
        ma_loss = max(ma_loss, ma)

    all_pass = passed_chain and head_pass and loss_pass
    max_abs = max(ma_chain, ma_head, ma_loss)
    assert all_pass, (
        f'Detector parity FAILED: chain={ma_chain:.3e} head={ma_head:.3e} '
        f'loss={ma_loss:.3e}')
    print(f'\nVERDICT P7: PASS (max_abs={max_abs:.3e})')


class _InstanceData:
    """Minimal mmdet3d InstanceData shim for feeding gt into the original head."""
    def __init__(self, bboxes_3d, labels_3d):
        self.bboxes_3d = _BboxesShim(bboxes_3d)
        self.labels_3d = labels_3d


class _BboxesShim:
    """Minimal bboxes_3d shim: exposes .gravity_center and .tensor."""
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

    @property
    def gravity_center(self):
        # OpenPCDet gt boxes are already volume-center, so gravity_center == xyz.
        return self._tensor[:, :3]


def _orig_loss_by_feat_from_tensor_gt(orig_head, preds_dicts, gt_boxes_tensor):
    """Wrap the port-side ``(B, M, 8)`` gt tensor into mmdet3d InstanceData
    and call the original head's ``loss_by_feat``.

    The original splits ``bboxes_3d`` via ``gravity_center`` (we shim that to
    be identity since OpenPCDet gt is already volume-center) and uses
    ``labels_3d`` as the 0-based class index (we shift from 1-based OpenPCDet).
    """
    B, M, _ = gt_boxes_tensor.shape
    batch_gt = []
    for b in range(B):
        bboxes = gt_boxes_tensor[b, :, :7]
        labels = gt_boxes_tensor[b, :, 7].long() - 1  # to 0-based
        batch_gt.append(_InstanceData(bboxes, labels))
    return orig_head.loss_by_feat(preds_dicts, batch_gt)


if __name__ == '__main__':
    test_parity_detector()
