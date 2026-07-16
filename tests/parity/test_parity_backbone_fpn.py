"""Parity point 3: end-to-end FPN backbone (RepDWC + SECONDFPN chained).

Port side: ``RadarNeXtFPNBackbone`` (one BACKBONE_2D wrapping both halves).
Original side: ``RepDWC`` and ``SECONDFPN`` chained manually (the original
mmdet3d detector wires them as ``self.backbone`` and ``self.neck`` siblings).

Both halves are weight-aligned, fed the same (B,32,320,320) input. The
fused (B, 384, 80, 80) output must be element-wise identical.

This is an integration check: since P1 and P2 already pass bit-for-bit,
this should also pass. The value is exercising the port's
``RadarNeXtFPNBackbone.forward`` (which wraps both halves and does the
``self.fpn(...)[0]`` indexing) against the original two-step chain.
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
    LOOSE_ATOL, LOOSE_RTOL,
)
from build_weight_map import build_backbone_fpn_both


def test_parity_backbone_fpn():
    seed_rng(0)
    port, (orig_backbone, orig_fpn) = build_backbone_fpn_both()

    # Weight alignment. The port wraps RepDWC under ``backbone.*`` and
    # SECONDFPN under ``fpn.*``; align each half to the corresponding original.
    port_sd = port.state_dict()
    ob_sd = orig_backbone.state_dict()
    of_sd = orig_fpn.state_dict()

    # Build a "virtual original" state dict with prefixes the port uses.
    merged_orig_sd = {}
    for k, v in ob_sd.items():
        merged_orig_sd[f'backbone.{k}'] = v
    for k, v in of_sd.items():
        merged_orig_sd[f'fpn.{k}'] = v

    matched, missed = 0, []
    new_sd = {}
    for k, v in port_sd.items():
        if k in merged_orig_sd and merged_orig_sd[k].shape == v.shape:
            new_sd[k] = merged_orig_sd[k].clone()
            matched += 1
        else:
            new_sd[k] = v
            missed.append(k)
    port.load_state_dict(new_sd, strict=False)
    print(f'  [align] matched={matched}/{len(port_sd)} missed={len(missed)}')
    assert not missed, f'unmatched port keys: {missed[:5]}'

    port.eval(); orig_backbone.eval(); orig_fpn.eval()
    x = gen_bev(batch=2, channels=C.REPDWC_IN_CHANNELS, h=320, w=320,
                seed=123)

    with torch.no_grad():
        # Port: single forward through the wrapper.
        dd = {'spatial_features': x}
        out_port = port(dd)['spatial_features_2d']
        # Original: two-step chain (backbone -> neck).
        ms = list(orig_backbone(x))
        out_orig = orig_fpn(ms)[0]

    print('\n--- RadarNeXtFPNBackbone output shapes ---')
    print(f'  port={tuple(out_port.shape)} orig={tuple(out_orig.shape)}')

    passed, max_abs, max_rel = parity_allclose(
        out_port, out_orig, atol=LOOSE_ATOL, rtol=LOOSE_RTOL,
        name='RadarNeXtFPNBackbone')
    assert passed, (
        f'RadarNeXtFPNBackbone parity FAILED: max_abs_diff={max_abs:.6e} '
        f'max_rel_diff={max_rel:.6e}')
    print(f'\nVERDICT P3: PASS (max_abs={max_abs:.3e})')


if __name__ == '__main__':
    test_parity_backbone_fpn()
