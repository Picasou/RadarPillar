"""DCNv3 op (pure-pytorch fallback) for OpenPCDet.

Provides ``DCNv3_pytorch`` (a faithful, grid_sample-based reimplementation
of InternImage's DCNv3 module that requires NO CUDA extension) so the
MDFEN neck (Task 7) can run in any environment.

Selection rationale (execution protocol §6 never-fail guarantee):
    The CUDA ``DCNv3`` op from InternImage's ``ops_dcnv3`` requires a
    compiled extension and is environment-sensitive. This module ships the
    pure-pytorch fallback path that the original RadarNeXt repo also keeps
    available (``DeformFFN.py``'s ``DCNv3_pytorch`` class). It is the
    §6 floor — runs anywhere torch runs.
"""

from .dcnv3_pytorch import (
    DCNv3_pytorch,
    dcnv3_core_pytorch,
    CenterFeatureScaleModule,
    build_norm_layer,
    build_act_layer,
)

__all__ = [
    'DCNv3_pytorch',
    'dcnv3_core_pytorch',
    'CenterFeatureScaleModule',
    'build_norm_layer',
    'build_act_layer',
]
