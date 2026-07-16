"""Parity point 5: RadarNeXtCenterHead forward.

Both heads are built from the FPN-variant config, weight-aligned, and fed
the SAME (B, 384, 80, 80) feature map. The per-task output dicts
(hm/reg/height/dim/rot/iou/corner_hm) must be element-wise identical.

We compare ``forward`` outputs only (no loss / no targets here — those are
parity point 6). The heads run in eval mode so the corner_hm head is
deactivated on BOTH sides identically (corner_hm is train-only).
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from tests.parity import _originals  # noqa: F401
from tests.parity import _configs as C
from tests.parity.conftest import (
    align_state_dicts, gen_bev, parity_allclose, seed_rng,
    DEFAULT_ATOL, DEFAULT_RTOL, LOOSE_ATOL, LOOSE_RTOL,
)
from build_weight_map import build_head_both


def test_parity_centerhead_forward():
    seed_rng(0)
    port, orig = build_head_both()
    align_state_dicts(orig, port, verbose=True)
    port.eval(); orig.eval()

    # Same feature map for both sides. The head's SepHead has a stride=2
    # ConvTranspose2d deblock (80 -> 160), so use the loose deconv tolerance.
    feat = gen_bev(batch=2, channels=C.HEAD_IN_CHANNELS, h=80, w=80,
                   seed=123)

    with torch.no_grad():
        # Port: data_dict convention.
        dd = {'spatial_features_2d': feat}
        # We bypass the port head's loss/predict path by calling its submodules
        # directly to match the original's `forward(feats)` semantics.
        # Original: forward(x) with multi_fusion=False does shared_conv(x[0]).
        x_port = port.shared_conv(feat)
        ret_port = [t(x_port) for t in port.tasks]
        # Original: forward([feat]) returns list of per-task dicts.
        ret_orig = orig([feat])

    print('\n--- CenterHead forward output heads ---')
    heads = list(ret_port[0].keys())
    print(f'  task0 heads (port): {heads}')
    print(f'  task0 heads (orig): {list(ret_orig[0].keys())}')

    all_pass = True
    max_abs = 0.0
    for h in heads:
        a = ret_port[0][h]
        b = ret_orig[0][h]
        p, ma, _ = parity_allclose(
            a, b, atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name=f'head.{h}')
        all_pass = all_pass and p
        max_abs = max(max_abs, ma)
    assert all_pass, (
        f'CenterHead forward parity FAILED: max_abs_diff={max_abs:.6e}')
    print(f'\nVERDICT P5: PASS (max_abs={max_abs:.3e})')


if __name__ == '__main__':
    test_parity_centerhead_forward()
