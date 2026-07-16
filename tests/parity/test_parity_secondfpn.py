"""Parity point 2: SECONDFPN neck.

Both sides are fed the SAME multi-scale inputs (the 3 scales from a
weight-aligned RepDWC, so the FPN sees real, matching features — not
independent random tensors). Output (B, 384, 80, 80) must be element-wise
identical.

This also serves as a smoke test for the deconv/conv selection logic
(``upsample_strides=[0.5, 1, 2]``: 0.5 -> Conv2d k=2 s=2; 1 -> Conv2d k=1
because ``use_conv_for_no_stride=True``; 2 -> ConvTranspose2d).
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
from build_weight_map import build_repdwc_both, build_secondfpn_both


def test_parity_secondfpn():
    seed_rng(0)
    port_fpn, orig_fpn = build_secondfpn_both()
    align_state_dicts(orig_fpn, port_fpn, verbose=True)
    port_fpn.eval(); orig_fpn.eval()

    # Use RepDWC (weight-aligned) outputs as the multi-scale input so the FPN
    # sees real, matching features rather than two independent random tensors.
    port_backbone, orig_backbone = build_repdwc_both()
    align_state_dicts(orig_backbone, port_backbone, verbose=False)
    port_backbone.eval(); orig_backbone.eval()
    x = gen_bev(batch=2, channels=C.REPDWC_IN_CHANNELS, h=320, w=320,
                seed=123)
    with torch.no_grad():
        ms_port = port_backbone(x)
        ms_orig = list(orig_backbone(x))
    # sanity: the upstream RepDWC parity already passed, so ms_port == ms_orig.

    with torch.no_grad():
        out_port = port_fpn(ms_port)[0]
        out_orig = orig_fpn(ms_orig)[0]

    print('\n--- SECONDFPN output shapes ---')
    print(f'  port={tuple(out_port.shape)} orig={tuple(out_orig.shape)}')

    # SecondFPN uses ConvTranspose2d (deconv); allow the looser tolerance.
    passed, max_abs, max_rel = parity_allclose(
        out_port, out_orig,
        atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name='SECONDFPN')
    assert passed, (
        f'SECONDFPN parity FAILED: max_abs_diff={max_abs:.6e} '
        f'max_rel_diff={max_rel:.6e}')
    print(f'\nVERDICT P2: PASS (max_abs={max_abs:.3e})')


if __name__ == '__main__':
    test_parity_secondfpn()
