"""Parity point 4: MDFENNeck (the deferred point from Task 4.5).

This is the main verification of the Task 7 MDFEN port. Synthetic 3-scale
inputs are fed to BOTH the port's ``MDFENNeck`` (in
``pcdet/models/backbones_2d/mdfen_neck.py``) and the ORIGINAL RadarNeXt
``MDFENNeck`` (from ``projects/RadarNeXt/radarnext/MDFENNeck.py``) with
COPIED weights, then the fused ``(B, 384, 80, 80)`` outputs are compared
with ``atol=1e-3`` (loose, because the path runs DCNv3 + grid_sample +
deconv arithmetic).

DCNv3 handling (per Task 7 brief, §6 never-fail):
    The original's ``common.DeformLayer`` uses a bare ``DCNv3`` (since
    ``use_ffn=False``). The parity loader
    (``tests.parity._originals.load_mdfen_originals``) patches BOTH the
    port's ``DCNv3_pytorch`` AND the original's ``DCNv3`` to point at the
    SAME ``pcdet.ops.dcnv3.DCNNv3_pytorch`` class, so any divergence is
    attributable to the neck wrapping (PAN bidirectional flow, RepBlock
    stacking, MultiMAPFusion), NOT to a DCNv3 implementation difference.

Weight alignment:
    Both modules are seeded identically at construction, then the original's
    state_dict is copied into the port by exact key+shape match
    (``align_state_dicts``). The port keeps the SAME submodule names as the
    original (``reduce_layer0``, ``Rep_p4``, ``former_deform2``,
    ``fusion.blocks`` etc.) so the map is identity.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from tests.parity import _originals  # noqa: F401  (installs stubs)
from tests.parity.conftest import (
    align_state_dicts, gen_multiscale, parity_allclose, seed_rng,
    LOOSE_ATOL, LOOSE_RTOL,
)
from pcdet.models.backbones_2d.mdfen_neck import MDFENNeck as PortMDFEN


# Production config (projects/RadarNeXt/configs/radarnext.py 'neck' block).
MDFEN_KWARGS = dict(
    channels_list=[64, 128, 256, 128, 64, 128, 256],
    num_repeats=[1, 1, 1, 1],
    dcn_layer=False,
    dcn_index=[1],
    former=True,
    latter=False,
    dcn_ids=[2],
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
)


def _build_port_and_orig():
    """Build the port + original MDFENNeck with identical RNG seed."""
    seed_rng(0)
    port = PortMDFEN(**MDFEN_KWARGS)
    orig_mods = _originals.load_mdfen_originals()
    OrigMDFEN = orig_mods['MDFENNeck'].MDFENNeck
    seed_rng(0)
    orig = OrigMDFEN(**MDFEN_KWARGS)
    return port, orig


def test_parity_mdfen_neck_fused_output():
    """P4: port MDFENNeck vs original MDFENNeck fused output (B,384,80,80)."""
    port, orig = _build_port_and_orig()
    align_state_dicts(orig, port, verbose=True)
    port.eval()
    orig.eval()

    # Synthetic 3-scale inputs (x2, x1, x0): x2 = largest spatial + 64ch,
    # x0 = smallest spatial + 256ch (matches the mmdet3d PAN input contract).
    shapes = [
        (2, 64, 160, 160),    # x2 (largest)
        (2, 128, 80, 80),     # x1
        (2, 256, 40, 40),     # x0 (smallest)
    ]
    inputs = gen_multiscale(shapes, seed=123)

    with torch.no_grad():
        out_port = port(list(inputs))[0]
        out_orig = orig(list(inputs))[0]

    print('\n--- MDFENNeck fused output shapes ---')
    print(f'  port={tuple(out_port.shape)} orig={tuple(out_orig.shape)}')

    passed, max_abs, max_rel = parity_allclose(
        out_port, out_orig,
        atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name='MDFENNeck_fused')
    assert passed, (
        f'MDFENNeck parity FAILED: max_abs_diff={max_abs:.6e} '
        f'max_rel_diff={max_rel:.6e}')
    print(f'\nVERDICT P4: PASS (max_abs={max_abs:.3e})')


def test_parity_mdfen_neck_multiscale_intermediate():
    """P4b (bonus): multi-scale PAN outputs (multi_fusion disabled) match too.

    Sets ``multi_fusion=False`` so the neck returns the raw 3-scale PAN
    outputs ``[pan_out2, pan_out1, pan_out0]``; compares each scale. This
    isolates the PAN bidirectional flow (incl. the ``former_deform2`` DCN
    site) from the MultiMAPFusion aggregation.
    """
    kwargs_no_fusion = dict(MDFEN_KWARGS)
    kwargs_no_fusion['multi_fusion'] = False

    seed_rng(0)
    port = PortMDFEN(**kwargs_no_fusion)
    orig_mods = _originals.load_mdfen_originals()
    OrigMDFEN = orig_mods['MDFENNeck'].MDFENNeck
    seed_rng(0)
    orig = OrigMDFEN(**kwargs_no_fusion)
    align_state_dicts(orig, port, verbose=False)
    port.eval()
    orig.eval()

    shapes = [
        (2, 64, 160, 160),
        (2, 128, 80, 80),
        (2, 256, 40, 40),
    ]
    inputs = gen_multiscale(shapes, seed=321)

    with torch.no_grad():
        outs_port = port(list(inputs))
        outs_orig = orig(list(inputs))

    assert len(outs_port) == len(outs_orig) == 3, \
        f'expected 3 PAN outputs, got port={len(outs_port)} orig={len(outs_orig)}'

    print('\n--- MDFENNeck multi-scale PAN outputs ---')
    all_pass = True
    worst_abs = 0.0
    names = ['pan_out2 (160x160, 64ch)', 'pan_out1 (80x80, 128ch)',
             'pan_out0 (40x40, 256ch)']
    for i, (a, b, name) in enumerate(zip(outs_port, outs_orig, names)):
        p, ma, mr = parity_allclose(
            a, b, atol=LOOSE_ATOL, rtol=LOOSE_RTOL, name=f'P4b[{i}] {name}')
        all_pass = all_pass and p
        worst_abs = max(worst_abs, ma)
    assert all_pass, (
        f'MDFENNeck multi-scale parity FAILED: worst max_abs={worst_abs:.6e}')
    print(f'\nVERDICT P4b: PASS (worst max_abs={worst_abs:.3e})')


def test_parity_mdfen_param_count():
    """P4c (sanity): port and original MDFENNeck have identical param count."""
    port, orig = _build_port_and_orig()
    n_port = sum(p.numel() for p in port.parameters())
    n_orig = sum(p.numel() for p in orig.parameters())
    print(f'\n--- MDFENNeck param counts ---')
    print(f'  port={n_port:,}  orig={n_orig:,}')
    assert n_port == n_orig, (
        f'param count mismatch: port={n_port} vs orig={n_orig}')
    print(f'\nVERDICT P4c: PASS (params identical: {n_port:,})')


if __name__ == '__main__':
    test_parity_mdfen_param_count()
    test_parity_mdfen_neck_multiscale_intermediate()
    test_parity_mdfen_neck_fused_output()
