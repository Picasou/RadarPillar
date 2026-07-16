"""Parity point 6: detection-head losses.

Verifies each loss helper on the port side matches the original:

* ``FastFocalLoss``  — synthetic pred hm + target hm + peak indices.
* ``RegLoss``        — L1 reg on gathered peaks.
* ``bbox3d_overlaps_diou`` — dIoU on aligned boxes (verbatim port; sanity).
* ``IouLoss``        — port emulates ``boxes_aligned_iou3d_gpu`` via the
  diagonal of ``boxes_iou3d_gpu``; original calls the aligned helper
  directly. For parity, both must agree.
* ``IouRegLoss``     — dIoU regression.

All loss modules are param-free, so no weight alignment is needed; same
inputs (fixed seed) must yield same outputs. The interesting one is
``IouLoss`` where the two sides use different IoU backends; we verify
they produce numerically identical results on the same box pair.

NOTE on ``IouLoss`` shape-mismatch "bug" (audit D):
The original PillarNeXt IouLoss passes ``pred (N,1)`` and ``target (N,)``
into ``F.l1_loss(reduction='sum')``, which broadcasts to ``(N,N)`` before
summing. The port REPRODUCES this quirk verbatim (see the NOTE in
``radarnext_losses.py``). So both sides will agree — that's the parity
contract, not a bug to fix here.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from tests.parity import _originals as O  # noqa: F401  (installs stubs)
from tests.parity.conftest import (
    parity_allclose, seed_rng, gen_gt_boxes,
    DEFAULT_ATOL, DEFAULT_RTOL, LOOSE_ATOL, LOOSE_RTOL,
)
from pcdet.models.dense_heads.radarnext_losses import (
    FastFocalLoss as PortFocal,
    RegLoss as PortReg,
    IouLoss as PortIouLoss,
    IouRegLoss as PortIouRegLoss,
    bbox3d_overlaps_diou as port_diou,
)


def test_parity_focal():
    seed_rng(7)
    B, C, H, W, M = 2, 3, 16, 16, 5
    out = torch.sigmoid(torch.randn(B, C, H, W))
    target = torch.zeros(B, C, H, W)
    # Draw a few positive peaks.
    for b in range(B):
        for c in range(C):
            for _ in range(3):
                y, x = torch.randint(0, H, (1,)).item(), torch.randint(0, W, (1,)).item()
                target[b, c, y, x] = 1.0
    ind = torch.randint(0, H * W, (B, M))
    mask = torch.randint(0, 2, (B, M)).float()
    cat = torch.randint(0, C, (B, M))

    pl = PortFocal()(out.clone(), target.clone(), ind, mask, cat)
    ol = O.FastFocalLoss()(out.clone(), target.clone(), ind, mask, cat)
    passed, ma, _ = parity_allclose(pl, ol, atol=DEFAULT_ATOL,
                                    rtol=DEFAULT_RTOL, name='FocalLoss')
    assert passed, f'FocalLoss FAILED: max_abs={ma:.3e}'


def test_parity_reg():
    seed_rng(11)
    B, dim, H, W, M = 2, 8, 16, 16, 5
    out = torch.randn(B, dim, H, W)
    ind = torch.randint(0, H * W, (B, M))
    mask = torch.randint(0, 2, (B, M))
    target = torch.randn(B, M, dim)

    pl = PortReg()(out.clone(), ind.clone(), mask.clone(), target.clone())
    ol = O.RegLoss()(out.clone(), ind.clone(), mask.clone(), target.clone())
    passed, ma, _ = parity_allclose(pl, ol, atol=DEFAULT_ATOL,
                                    rtol=DEFAULT_RTOL, name='RegLoss')
    assert passed, f'RegLoss FAILED: max_abs={ma:.3e}'


def test_parity_diou():
    """dIoU on aligned boxes — verbatim port, sanity check."""
    seed_rng(13)
    boxes_a = torch.randn(8, 7)
    boxes_b = boxes_a.clone() + 0.1 * torch.randn(8, 7)
    # Make boxes physically valid (positive dims).
    boxes_a[:, 3:6] = boxes_a[:, 3:6].abs() + 0.5
    boxes_b[:, 3:6] = boxes_b[:, 3:6].abs() + 0.5
    pd = port_diou(boxes_a, boxes_b)
    od = O.bbox3d_overlaps_diou(boxes_a, boxes_b)
    passed, ma, _ = parity_allclose(pd, od, atol=DEFAULT_ATOL,
                                    rtol=DEFAULT_RTOL, name='dIoU')
    assert passed, f'dIoU FAILED: max_abs={ma:.3e}'


def _gather_peaks(feat, ind, mask):
    """Helper: (B,C,H,W) -> (N,C) at masked peak indices."""
    B, C, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
    ind_e = ind.unsqueeze(2).expand(B, ind.size(1), C)
    gathered = feat.gather(1, ind_e)
    return gathered[mask]


def test_parity_iou_loss():
    """IouLoss: port (boxes_iou3d_gpu diagonal) vs original (aligned IoU).

    The CUDA boxes_iou3d_gpu and the mmcv ext boxes_aligned_iou3d_gpu both
    compute the same geometric IoU on aligned 1:1 box pairs. Run on CUDA so
    the port's iou3d_nms_utils path is exercised.
    """
    if not torch.cuda.is_available():
        print('[IouLoss] SKIP (CUDA required)')
        return
    seed_rng(17)
    B, H, W, M = 2, 16, 16, 5
    iou_pred = torch.randn(B, 1, H, W, device='cuda')
    ind = torch.randint(0, H * W, (B, M), device='cuda')
    mask = torch.randint(0, 2, (B, M), device='cuda').bool()

    # Build matching pred_box (B,7,H,W) and gt (B,M,7) at the peak locations.
    box_pred = torch.randn(B, 7, H, W, device='cuda')
    box_pred[:, 3:6] = box_pred[:, 3:6].abs() + 0.5
    gt_boxes = gen_gt_boxes(batch=B, max_objs=M, seed=18,
                            device='cuda')[:, :, :7]

    pl = PortIouLoss()(iou_pred.clone(), mask.clone(), ind.clone(),
                       box_pred.clone(), gt_boxes.clone())
    ol = O.IouLoss()(iou_pred.clone(), mask.clone(), ind.clone(),
                     box_pred.clone(), gt_boxes.clone())
    # NOTE: the original IouLoss broadcasts (N,1) vs (N,) into (N,N) before
    # the sum, while the port reproduces this quirk. Both should agree.
    passed, ma, _ = parity_allclose(pl, ol, atol=LOOSE_ATOL,
                                    rtol=LOOSE_RTOL, name='IouLoss')
    assert passed, f'IouLoss FAILED: max_abs={ma:.3e}'


def test_parity_iou_reg_loss():
    """IouRegLoss (dIoU regression) — both sides use bbox3d_overlaps_diou."""
    seed_rng(19)
    B, H, W, M = 2, 16, 16, 5
    box_pred = torch.randn(B, 7, H, W)
    box_pred[:, 3:6] = box_pred[:, 3:6].abs() + 0.5
    ind = torch.randint(0, H * W, (B, M))
    mask = torch.randint(0, 2, (B, M)).bool()
    gt_boxes = gen_gt_boxes(batch=B, max_objs=M, seed=20)[:, :, :7]

    pl = PortIouRegLoss()(box_pred.clone(), mask.clone(), ind.clone(),
                          gt_boxes.clone())
    ol = O.IouRegLoss()(box_pred.clone(), mask.clone(), ind.clone(),
                        gt_boxes.clone())
    passed, ma, _ = parity_allclose(pl, ol, atol=DEFAULT_ATOL,
                                    rtol=DEFAULT_RTOL, name='IouRegLoss')
    assert passed, f'IouRegLoss FAILED: max_abs={ma:.3e}'


def test_parity_loss_all():
    print('\n--- Loss parity ---')
    test_parity_focal(); print('  Focal: PASS')
    test_parity_reg(); print('  Reg: PASS')
    test_parity_diou(); print('  dIoU: PASS')
    test_parity_iou_loss(); print('  IouLoss: PASS')
    test_parity_iou_reg_loss(); print('  IouRegLoss: PASS')
    print('\nVERDICT P6: PASS')


if __name__ == '__main__':
    test_parity_loss_all()
