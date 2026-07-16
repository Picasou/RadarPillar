"""Parity point 1: RepDWC backbone.

Both sides are built from the FPN-variant config (in_channels=32 per M4
audit, out=[64,128,256], layer_nums=[3,5,5], strides=[2,2,2]). They are
seeded with ``torch.manual_seed(0)`` AND weight-aligned by name (premise #1)
before the same (B,32,320,320) input is fed. The 3 multi-scale outputs must
be element-wise identical.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from tests.parity import _originals  # noqa: F401  (installs stubs)
from tests.parity import _configs as C
from tests.parity.conftest import (
    align_state_dicts, gen_bev, parity_allclose_list, seed_rng,
    DEFAULT_ATOL, DEFAULT_RTOL,
)
from build_weight_map import build_repdwc_both


def test_parity_repdwc():
    seed_rng(0)
    port, orig = build_repdwc_both()

    # Premise #1 — weight alignment. The port and original use identical
    # submodule names (``blocks.*``), so the map is identity.
    report = align_state_dicts(orig, port, verbose=True)
    assert not report['shape_mismatch'], \
        f'shape mismatch: {report["shape_mismatch"][:3]}'
    # Some buffers/conv biases may be in unmatched_dst if names ever diverge;
    # for this module we expect exact 1:1 matching.
    assert len(report['unmatched_src']) == 0, \
        f'unmatched src keys: {report["unmatched_src"][:3]}'

    port.eval(); orig.eval()
    # Premise #2 — same fixed-seed input.
    x = gen_bev(batch=2, channels=C.REPDWC_IN_CHANNELS, h=320, w=320,
                seed=123)

    with torch.no_grad():
        out_port = port(x)
        out_orig = orig(x)
        # Original returns tuple; port returns list. Normalize.
        out_orig = list(out_orig)

    print('\n--- RepDWC output shapes ---')
    for i, (a, b) in enumerate(zip(out_port, out_orig)):
        print(f'  scale[{i}]: port={tuple(a.shape)} orig={tuple(b.shape)}')

    passed, max_abs, max_rel = parity_allclose_list(
        out_port, out_orig, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL,
        name='RepDWC')
    assert passed, (
        f'RepDWC parity FAILED: max_abs_diff={max_abs:.6e} '
        f'max_rel_diff={max_rel:.6e}')
    print(f'\nVERDICT P1: PASS (max_abs={max_abs:.3e})')


if __name__ == '__main__':
    test_parity_repdwc()
