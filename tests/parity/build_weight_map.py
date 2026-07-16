"""Build layer-name maps between the port and the RadarNeXt original.

For the FPN-chain modules, the port keeps the SAME submodule names as the
original (``blocks``, ``deblocks``, ``shared_conv``, ``tasks``). So the
weight map is *identity* — both sides' ``state_dict().keys()`` should match
exactly. This script verifies that and emits a JSON map per module.

If a future port renames submodules, edit ``MANUAL_OVERRIDES`` below.

Usage::

    python tests/parity/build_weight_map.py [--out-dir DIR]

Emits ``weight_map_<module>.json`` files (default: ``tests/parity/maps/``).
"""

import argparse
import json
import os
import sys

# Make the project root importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from easydict import EasyDict  # noqa: E402

from tests.parity import _originals  # noqa: E402  (installs stubs)
from tests.parity import _configs as C  # noqa: E402
from tests.parity.conftest import seed_rng  # noqa: E402

# Port modules.
from pcdet.models.backbones_2d.rep_dwc import RepDWCBackbone  # noqa: E402
from pcdet.models.backbones_2d.second_fpn import SecondFPN  # noqa: E402
from pcdet.models.backbones_2d.radarnext_backbone_fpn import (  # noqa: E402
    RadarNeXtFPNBackbone,
)
from pcdet.models.dense_heads.radarnext_center_head import (  # noqa: E402
    RadarNeXtCenterHead,
)


# --------------------------------------------------------------------------- #
# Constructors (both sides, seeded identically)                                #
# --------------------------------------------------------------------------- #
def build_repdwc_both():
    seed_rng(0)
    port = RepDWCBackbone(model_cfg=C.build_repdwc_cfg_port(),
                          input_channels=C.REPDWC_IN_CHANNELS)
    seed_rng(0)
    orig = _originals.RepDWC(**C.build_repdwc_kwargs_orig())
    return port, orig


def build_secondfpn_both():
    seed_rng(0)
    port = SecondFPN(
        in_channels=list(C.FPN_IN_CHANNELS),
        out_channels=list(C.FPN_OUT_CHANNELS),
        upsample_strides=list(C.FPN_UPSAMPLE_STRIDES),
        norm_cfg=dict(C.FPN_NORM_CFG),
        upsample_cfg=dict(C.FPN_UPSAMPLE_CFG),
        conv_cfg=dict(C.FPN_CONV_CFG),
        use_conv_for_no_stride=C.FPN_USE_CONV_FOR_NO_STRIDE,
    )
    seed_rng(0)
    orig = _originals.SECONDFPN(**C.build_secondfpn_kwargs_orig())
    return port, orig


def build_backbone_fpn_both():
    seed_rng(0)
    port = RadarNeXtFPNBackbone(model_cfg=C.build_backbone_fpn_cfg_port(),
                                input_channels=C.REPDWC_IN_CHANNELS)
    # The original "RadarNeXt backbone+neck" combo is just RepDWC + SECONDFPN
    # as two siblings — we compare the two halves separately, so no monolithic
    # original to build here. The port-side RadarNeXtFPNBackbone map covers
    # both halves with prefixes backbone.* and fpn.*.
    seed_rng(0)
    orig_repdwc = _originals.RepDWC(**C.build_repdwc_kwargs_orig())
    seed_rng(0)
    orig_fpn = _originals.SECONDFPN(**C.build_secondfpn_kwargs_orig())
    return port, (orig_repdwc, orig_fpn)


def build_head_both():
    seed_rng(0)
    port = RadarNeXtCenterHead(
        model_cfg=C.build_head_cfg_port(),
        input_channels=C.HEAD_IN_CHANNELS,
        num_class=3,
        class_names=C.CLASS_NAMES_PORT,
        grid_size=C.GRID_SIZE,
        point_cloud_range=C.POINT_CLOUD_RANGE,
        predict_boxes_when_training=True,
    )
    seed_rng(0)
    orig = _originals.RadarNeXt_Head(**C.build_head_kwargs_orig())
    return port, orig


# --------------------------------------------------------------------------- #
# Map construction                                                             #
# --------------------------------------------------------------------------- #
def keymap_identity(port_sd, orig_sd):
    """Identity map (key -> key). Records shape-match status."""
    m = {}
    for k in port_sd:
        if k in orig_sd:
            m[k] = {
                'orig_key': k,
                'shape_match': tuple(port_sd[k].shape)
                                == tuple(orig_sd[k].shape),
                'port_shape': tuple(port_sd[k].shape),
                'orig_shape': tuple(orig_sd[k].shape),
            }
        else:
            m[k] = {'orig_key': None, 'shape_match': False,
                    'port_shape': tuple(port_sd[k].shape),
                    'orig_shape': None}
    return m


def summary(m):
    matched = sum(1 for v in m.values() if v['shape_match'])
    missing = sum(1 for v in m.values() if v['orig_key'] is None)
    shape_mm = sum(1 for v in m.values()
                   if v['orig_key'] is not None and not v['shape_match'])
    return dict(total=len(m), matched=matched,
                missing_in_orig=missing, shape_mismatch=shape_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.join(_HERE, 'maps'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    print('build_weight_map')
    print('=' * 60)

    plans = [
        ('repdwc', build_repdwc_both),
        ('secondfpn', build_secondfpn_both),
        ('centerhead', build_head_both),
    ]
    for name, builder in plans:
        port, orig = builder()
        port_sd = port.state_dict()
        if isinstance(orig, tuple):
            # Backbone-FPN port vs (orig_repdwc, orig_fpn): merge originals' SDs
            # with prefixes the port uses.
            orig_repdwc, orig_fpn = orig
            merged = {}
            for k, v in orig_repdwc.state_dict().items():
                merged[f'backbone.{k}'] = v
            for k, v in orig_fpn.state_dict().items():
                merged[f'fpn.{k}'] = v
            orig_sd = merged
        else:
            orig_sd = orig.state_dict()
        m = keymap_identity(port_sd, orig_sd)
        s = summary(m)
        out = os.path.join(args.out_dir, f'weight_map_{name}.json')
        with open(out, 'w') as f:
            json.dump({'module': name, 'summary': s, 'map': m}, f, indent=2)
        print(f'[{name}] {s} -> {out}')
        if s['shape_mismatch'] or s['missing_in_orig']:
            for k, v in m.items():
                if v['orig_key'] is None or not v['shape_match']:
                    print(f'    ! {k}: port={v["port_shape"]} '
                          f'orig={v["orig_shape"]}')


if __name__ == '__main__':
    main()
