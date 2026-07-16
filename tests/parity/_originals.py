"""Shim that exposes the RadarNeXt ORIGINAL modules for parity testing.

This is the right-hand side of every parity test. It installs pure-torch
stubs for the mmdet3d / mmengine / mmcv machinery that the original code
imports at module-load time (so we never need the DCNv3 CUDA op or the
mmdet3d/mmengine Python packages), then re-exports the originals.

Strategy
--------
The RadarNeXt originals (``projects/RadarNeXt/radarnext/rep_dwc.py`` etc.)
subclass ``mmengine.model.BaseModule`` and use the ``@MODELS.register_module``
decorator. Neither needs to *actually* exist for our purposes:

* ``BaseModule`` is replaced by a thin ``nn.Module`` subclass that ignores
  ``init_cfg`` — parity tests seed both sides with the same RNG state, so the
  mmdet3d init_cfg machinery would be irrelevant anyway.
* ``MODELS.register_module`` is replaced by a no-op decorator; we never call
  ``MODELS.build`` (we instantiate the originals directly).

The only hard external the originals pull in beyond the registry is
``DCNv3`` (via ``common.py`` -> ``DeformFFN``), and the mmcv CUDA ext
(``box_torch_ops`` -> ``ext_loader.load_ext('iou3d_nms3d_forward')``). Both
are stubbed:

* ``DeformFFN`` module -> placeholder classes (parity point 4 MDFENNeck is
  DEFERRED to Task 7, so we never instantiate them).
* ``box_torch_ops.rotate_nms_pcdet`` and ``iou3d_nms_utils.boxes_iou3d_gpu``
  -> pure-torch fallbacks using shapely-based rotated-BEV overlap. These are
  only used by the head's IouLoss (1:1 aligned IoU; we use the diagonal) and
  IouRegLoss — both of which the port reproduces verbatim using the pcdet
  ops, so the *parity comparison* is well-defined.

Public API (after ``install_stubs()`` + ``load_originals()``):
    RepDWC            — original RepDWC backbone.
    SECONDFPN         — original mmdet3d SECONDFPN neck.
    RadarNeXt_Head    — original detection head.
    FastFocalLoss, RegLoss, IouLoss, IouRegLoss, bbox3d_overlaps_diou
                      — original loss helpers.
    MultiMAPFusion, SepHead — original head building blocks.
"""

import sys

from . import _canary

_canary.install_stubs()

# Now actually import the originals.
import importlib
import importlib.util

_rep_dwc = importlib.import_module('projects.RadarNeXt.radarnext.rep_dwc')
_common = importlib.import_module('projects.RadarNeXt.radarnext.common')
_head = importlib.import_module('projects.RadarNeXt.radarnext.radarnext_head')
_pnloss = importlib.import_module('projects.PillarNeXt.pillarnext.loss')
_pnconv = importlib.import_module('projects.PillarNeXt.pillarnext.utils.conv')

# mmdet3d is stubbed, so import second_fpn.py directly from its file.
RN_ROOT = '/home/admin/projects/RadarNeXt'
_spec = importlib.util.spec_from_file_location(
    'mmdet3d.models.necks.second_fpn',
    RN_ROOT + '/mmdet3d/models/necks/second_fpn.py')
_second_fpn = importlib.util.module_from_spec(_spec)
sys.modules['mmdet3d.models.necks.second_fpn'] = _second_fpn
_spec.loader.exec_module(_second_fpn)

RepDWC = _rep_dwc.RepDWC
SECONDFPN = _second_fpn.SECONDFPN
RadarNeXt_Head = _head.RadarNeXt_Head
SepHead_orig = _head.SepHead
MultiMAPFusion = _head.MultiMAPFusion
ConvBlock_orig = _pnconv.ConvBlock

FastFocalLoss = _pnloss.FastFocalLoss
RegLoss = _pnloss.RegLoss
IouLoss = _pnloss.IouLoss
IouRegLoss = _pnloss.IouRegLoss
bbox3d_overlaps_diou = _pnloss.bbox3d_overlaps_diou
center_to_corner2d_orig = _pnloss.center_to_corner2d

__all__ = [
    'RepDWC', 'SECONDFPN', 'RadarNeXt_Head', 'SepHead_orig',
    'MultiMAPFusion', 'ConvBlock_orig',
    'FastFocalLoss', 'RegLoss', 'IouLoss', 'IouRegLoss',
    'bbox3d_overlaps_diou', 'center_to_corner2d_orig',
    'load_mdfen_originals',
]


# --------------------------------------------------------------------------- #
# Task 7 — original MDFEN modules with REAL DCNv3_pytorch                      #
# --------------------------------------------------------------------------- #
def load_mdfen_originals():
    """Load the original RadarNeXt ``DeformFFN`` / ``common`` / ``MDFENNeck``
    modules with the placeholder DCNv3 symbols replaced by the port's REAL
    ``pcdet.ops.dcnv3.DCNv3_pytorch``.

    Both sides of the MDFEN parity comparison then run the SAME working
    DCNv3 op, so any output divergence is attributable to the neck wrapping
    (DeformLayer, MultiMAPFusion, PAN bidirectional flow) and NOT to a
    DCNv3 implementation difference.

    Returns a dict with keys:
        ``DeformFFN``  — original DeformFFN.py module (real DCNv3 patched in).
        ``common``     — original common.py module (real DCNv3 patched in).
        ``MDFENNeck``  — original MDFENNeck.py module.
        ``DCNv3``      — the DCNv3_pytorch class the original sees.
    """
    import importlib as _il
    import importlib.util as _ilu

    # 'import DCNv3 as Dv3' inside DeformFFN.py — stub it (only the CUDA
    # DCNv3 class references Dv3; we never instantiate that one).
    if 'DCNv3' not in sys.modules:
        sys.modules['DCNv3'] = type(sys)('DCNv3')

    RN_DEFORMFFN = RN_ROOT + '/projects/RadarNeXt/radarnext/DeformFFN.py'
    _df_spec = _ilu.spec_from_file_location(
        'projects.RadarNeXt.radarnext.mdfen_real_DeformFFN', RN_DEFORMFFN)
    deformffn = _ilu.module_from_spec(_df_spec)
    sys.modules['projects.RadarNeXt.radarnext.mdfen_real_DeformFFN'] = deformffn
    _df_spec.loader.exec_module(deformffn)

    # Patch DCNv3 / DCNv3_pytorch symbols to point at our port's real impl.
    from pcdet.ops.dcnv3 import DCNv3_pytorch as _RealDCNv3
    deformffn.DCNv3_pytorch = _RealDCNv3
    deformffn.DCNv3 = _RealDCNv3  # original DeformLayer uses bare DCNv3

    # Reload common.py with the real DeformFFN reachable under the package
    # name it imports from (``from .DeformFFN import DCNv3, DCNv3_pytorch,
    # DeformFFN``). We temporarily swap the placeholder module out, then
    # restore it so the rest of the parity suite is unaffected.
    pkg = sys.modules['projects.RadarNeXt.radarnext']
    placeholder_DeformFFN = sys.modules.get(
        'projects.RadarNeXt.radarnext.DeformFFN')
    placeholder_common = sys.modules.get(
        'projects.RadarNeXt.radarnext.common')

    RN_COMMON = RN_ROOT + '/projects/RadarNeXt/radarnext/common.py'
    _cm_spec = _ilu.spec_from_file_location(
        'projects.RadarNeXt.radarnext.mdfen_real_common', RN_COMMON)
    common = _ilu.module_from_spec(_cm_spec)
    sys.modules['projects.RadarNeXt.radarnext.mdfen_real_common'] = common
    sys.modules['projects.RadarNeXt.radarnext.DeformFFN'] = deformffn
    try:
        _cm_spec.loader.exec_module(common)
    finally:
        if placeholder_DeformFFN is not None:
            sys.modules['projects.RadarNeXt.radarnext.DeformFFN'] = \
                placeholder_DeformFFN
    common.DCNv3 = _RealDCNv3
    common.DCNv3_pytorch = _RealDCNv3

    # Reload MDFENNeck.py with the real common + DeformFFN reachable.
    RN_MDFEN = RN_ROOT + '/projects/RadarNeXt/radarnext/MDFENNeck.py'
    _mn_spec = _ilu.spec_from_file_location(
        'projects.RadarNeXt.radarnext.mdfen_real_MDFENNeck', RN_MDFEN)
    mdfen = _ilu.module_from_spec(_mn_spec)
    sys.modules['projects.RadarNeXt.radarnext.mdfen_real_MDFENNeck'] = mdfen
    sys.modules['projects.RadarNeXt.radarnext.common'] = common
    sys.modules['projects.RadarNeXt.radarnext.DeformFFN'] = deformffn
    try:
        _mn_spec.loader.exec_module(mdfen)
    finally:
        if placeholder_common is not None:
            sys.modules['projects.RadarNeXt.radarnext.common'] = \
                placeholder_common
        if placeholder_DeformFFN is not None:
            sys.modules['projects.RadarNeXt.radarnext.DeformFFN'] = \
                placeholder_DeformFFN

    return {
        'DeformFFN': deformffn,
        'common': common,
        'MDFENNeck': mdfen,
        'DCNv3': _RealDCNv3,
    }
